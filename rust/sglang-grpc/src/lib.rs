pub mod bridge;
pub mod server;
pub mod tokenizer;

pub mod proto {
    tonic::include_proto!("sglang.runtime.v1");
}

use pyo3::prelude::*;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::{Notify, Semaphore};

use bridge::PyBridge;
use tokenizer::RustTokenizer;

/// Handle returned to Python that controls the running gRPC server.
#[pyclass]
struct GrpcServerHandle {
    shutdown: Arc<Notify>,
    join_handle: Option<std::thread::JoinHandle<()>>,
}

#[pymethods]
impl GrpcServerHandle {
    /// Gracefully shut down the gRPC server.
    fn shutdown(&mut self) {
        self.shutdown.notify_one();
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }

    /// Check if the server thread is still running.
    fn is_alive(&self) -> bool {
        self.join_handle
            .as_ref()
            .map_or(false, |h| !h.is_finished())
    }
}

/// Extract tokenizer path and context_len from the Python RuntimeHandle (one-time GIL).
fn extract_tokenizer_info(runtime_handle: &PyObject) -> (Option<String>, i32) {
    Python::with_gil(|py| {
        let tm = match runtime_handle.getattr(py, "tokenizer_manager") {
            Ok(tm) => tm,
            Err(_) => return (None, 0),
        };

        let model_path: Option<String> = tm
            .getattr(py, "model_path")
            .ok()
            .and_then(|v| v.extract(py).ok());

        let context_len: i32 = tm
            .getattr(py, "model_config")
            .ok()
            .and_then(|mc| mc.getattr(py, "context_len").ok())
            .and_then(|v| v.extract(py).ok())
            .unwrap_or(0);

        (model_path, context_len)
    })
}

/// Read `max_total_num_tokens` from the scheduler via the Python bridge's
/// `get_server_info()` method.  Returns `None` if the field is absent or the
/// call fails (caller should fall back to a safe static default).
fn read_max_total_num_tokens(runtime_handle: &PyObject) -> Option<usize> {
    let json_str: String = Python::with_gil(|py| {
        runtime_handle
            .call_method0(py, "get_server_info")
            .ok()
            .and_then(|v| v.extract::<String>(py).ok())
    })?;

    let v: serde_json::Value = serde_json::from_str(&json_str).ok()?;
    // scheduler_info is merged flat into the JSON by grpc_bridge.py
    v.get("max_total_num_tokens")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
}

/// Start the gRPC server in a background thread with its own Tokio runtime.
///
/// Args:
///     host: Bind address (e.g., "0.0.0.0")
///     port: Port number (e.g., 40000)
///     runtime_handle: Python RuntimeHandle object with submit_generate, submit_embed, abort, etc.
///
/// Returns:
///     GrpcServerHandle that can be used to shut down the server.
/// Start the native gRPC server in a background thread.
///
/// `max_prefill_tokens`:
///   - `None`  → auto-detect budget from scheduler's `max_total_num_tokens`
///   - `0`     → disable admission control entirely
///   - `N > 0` → use exactly N tokens as the prefill semaphore budget
#[pyfunction]
#[pyo3(signature = (host, port, runtime_handle, worker_threads=4, max_prefill_tokens=None))]
fn start_server(
    host: String,
    port: u16,
    runtime_handle: PyObject,
    worker_threads: usize,
    max_prefill_tokens: Option<usize>,
) -> PyResult<GrpcServerHandle> {
    let addr: SocketAddr = format!("{}:{}", host, port).parse().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid address: {}", e))
    })?;

    // Extract tokenizer info from Python (one-time GIL acquisition)
    let (model_path, context_len) = extract_tokenizer_info(&runtime_handle);

    // Attempt to load the Rust tokenizer
    let rust_tokenizer = model_path
        .as_deref()
        .and_then(|p| RustTokenizer::from_model_path(p, context_len));

    // Resolve the prefill token budget:
    //   0        → disabled
    //   Some(n)  → explicit override
    //   None     → auto-detect from scheduler's KV cache capacity
    //
    // Known limitation — this semaphore is a *concurrency* limiter, not a
    // *rate* limiter.  It bounds how many token-worth of requests can be in
    // prefill simultaneously, but does not smooth the submission rate to match
    // the scheduler's throughput.  Under heavy overload (arriving rate >>
    // serving rate) the prefill queue still builds inside the Python scheduler,
    // inflating TTFT regardless of the budget size.  Benchmarks show that
    // halving the budget (e.g. 10 % of KV) produces the same TTFT as using
    // the full capacity — the bottleneck is rate, not concurrency.
    //
    // The legacy sgl-model-gateway achieves low TTFT (~200 ms) because its
    // worker connection pool acts as an implicit token bucket, drip-feeding
    // requests at exactly the rate the backend can absorb.  Reaching parity
    // requires a proper token-bucket rate limiter in front of the scheduler
    // submission path (e.g. the `governor` crate).  That is left as a
    // follow-up; the semaphore still provides a hard ceiling against KV-cache
    // OOM when individual requests carry very large max_new_tokens values.
    let budget: Option<usize> = match max_prefill_tokens {
        Some(0) => None,
        Some(n) => Some(n),
        None => {
            let detected = read_max_total_num_tokens(&runtime_handle);
            match detected {
                Some(b) => eprintln!(
                    "[sglang-grpc] prefill token budget auto-detected: {} tokens",
                    b
                ),
                None => eprintln!(
                    "[sglang-grpc] could not read max_total_num_tokens; \
                     prefill admission control disabled"
                ),
            }
            detected
        }
    };

    let semaphore = budget.map(|n| Arc::new(Semaphore::new(n)));
    let bridge = Arc::new(PyBridge::new(runtime_handle, rust_tokenizer, context_len, semaphore));
    let shutdown = Arc::new(Notify::new());
    let shutdown_clone = shutdown.clone();

    let join_handle = std::thread::Builder::new()
        .name("sglang-grpc".to_string())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(worker_threads)
                .enable_all()
                .thread_name("sglang-grpc-tokio")
                .build()
                .expect("Failed to build Tokio runtime for gRPC server");

            if let Err(e) = rt.block_on(server::run_grpc_server(addr, bridge, shutdown_clone)) {
                eprintln!("gRPC server error: {}", e);
            }
        })
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to spawn gRPC thread: {}",
                e
            ))
        })?;

    Ok(GrpcServerHandle {
        shutdown,
        join_handle: Some(join_handle),
    })
}

#[pymodule]
fn sglang_grpc_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start_server, m)?)?;
    m.add_class::<GrpcServerHandle>()?;
    Ok(())
}

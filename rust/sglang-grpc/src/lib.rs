pub mod bridge;
pub mod server;

pub mod proto {
    tonic::include_proto!("sglang.runtime.v1");
}

use pyo3::prelude::*;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Notify;

use bridge::PyBridge;

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

/// Start the gRPC server in a background thread with its own Tokio runtime.
///
/// Args:
///     host: Bind address (e.g., "0.0.0.0")
///     port: Port number (e.g., 40000)
///     runtime_handle: Python RuntimeHandle object with submit_generate, submit_embed, abort, etc.
///
/// Returns:
///     GrpcServerHandle that can be used to shut down the server.
#[pyfunction]
#[pyo3(signature = (host, port, runtime_handle))]
fn start_server(
    host: String,
    port: u16,
    runtime_handle: PyObject,
) -> PyResult<GrpcServerHandle> {
    let addr: SocketAddr = format!("{}:{}", host, port)
        .parse()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid address: {}", e)))?;

    let bridge = Arc::new(PyBridge::new(runtime_handle));
    let shutdown = Arc::new(Notify::new());
    let shutdown_clone = shutdown.clone();

    let join_handle = std::thread::Builder::new()
        .name("sglang-grpc".to_string())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(4)
                .enable_all()
                .thread_name("sglang-grpc-tokio")
                .build()
                .expect("Failed to build Tokio runtime for gRPC server");

            if let Err(e) = rt.block_on(server::run_grpc_server(addr, bridge, shutdown_clone)) {
                eprintln!("gRPC server error: {}", e);
            }
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to spawn gRPC thread: {}", e)))?;

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

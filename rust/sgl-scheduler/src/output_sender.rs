use crate::stream_output::serialize_and_enqueue_token_output;
use crate::types::OutputMessage;
use crate::zmq_sender::zmq_sender_loop;
use crossbeam_channel::{bounded, Sender};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::thread;

/// RustOutputSender sends serialized output batches to the detokenizer
/// via a background thread with a ZMQ PUSH socket.
///
/// The background thread owns the ZMQ socket and performs serialization
/// prefix + ZMQ send entirely off the GIL, so the scheduler loop is
/// not blocked by I/O.
#[pyclass]
pub struct RustOutputSender {
    tx: Sender<OutputMessage>,
    _thread: Option<thread::JoinHandle<()>>,
}

#[pymethods]
impl RustOutputSender {
    /// Create a new RustOutputSender.
    ///
    /// Args:
    ///     zmq_endpoint: The ZMQ endpoint to connect to (e.g., "ipc:///tmp/sglang-detok")
    ///     sndbuf_size: ZMQ send buffer size (-1 for default)
    #[new]
    fn new(zmq_endpoint: String, sndbuf_size: i32) -> PyResult<Self> {
        // Use a bounded channel to provide backpressure if the sender
        // falls behind. 128 is generous — in practice the scheduler
        // produces one message per forward pass.
        let (tx, rx) = bounded::<OutputMessage>(128);

        let endpoint = zmq_endpoint.clone();
        let handle = thread::Builder::new()
            .name("sgl-zmq-sender".into())
            .spawn(move || {
                zmq_sender_loop(rx, endpoint, sndbuf_size);
            })
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to spawn ZMQ sender thread: {}",
                    e
                ))
            })?;

        Ok(Self {
            tx,
            _thread: Some(handle),
        })
    }

    /// Send a pre-serialized token batch (msgpack bytes) to the background thread.
    /// The bytes are produced by Python-side msgpack serialization.
    fn send_token_batch_bytes(&self, data: Vec<u8>) -> PyResult<()> {
        self.tx
            .send(OutputMessage::TokenBatch(data))
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to enqueue token batch: {}",
                    e
                ))
            })
    }

    /// Send a pre-serialized embedding batch (msgpack bytes) to the background thread.
    fn send_embedding_batch_bytes(&self, data: Vec<u8>) -> PyResult<()> {
        self.tx
            .send(OutputMessage::EmbeddingBatch(data))
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to enqueue embedding batch: {}",
                    e
                ))
            })
    }

    /// Phase 2: Serialize a batch token output dict directly in Rust and send.
    ///
    /// Takes a Python dict with the same structure as BatchTokenIDOutput,
    /// serializes it to msgpack using rmp-serde (bypassing Python msgspec),
    /// and enqueues for background ZMQ send.
    fn serialize_and_send_token_output(&self, data: &Bound<'_, PyDict>) -> PyResult<()> {
        serialize_and_enqueue_token_output(&self.tx, data)
    }

    /// Shut down the background sender thread.
    fn shutdown(&self) -> PyResult<()> {
        let _ = self.tx.send(OutputMessage::Shutdown);
        Ok(())
    }
}

impl Drop for RustOutputSender {
    fn drop(&mut self) {
        let _ = self.tx.send(OutputMessage::Shutdown);
        if let Some(handle) = self._thread.take() {
            let _ = handle.join();
        }
    }
}

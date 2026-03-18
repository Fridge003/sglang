use pyo3::prelude::*;

mod output_sender;
mod serialization;
mod types;
mod zmq_sender;

use output_sender::RustOutputSender;

/// Python module for the sgl-scheduler Rust extension.
#[pymodule]
fn sgl_scheduler(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustOutputSender>()?;
    // Export format prefix constants for Python-side dual-format receive
    m.add("FORMAT_PREFIX_PICKLE", types::FORMAT_PREFIX_PICKLE)?;
    m.add("FORMAT_PREFIX_MSGPACK_TOKEN", types::FORMAT_PREFIX_MSGPACK_TOKEN)?;
    m.add("FORMAT_PREFIX_MSGPACK_EMBEDDING", types::FORMAT_PREFIX_MSGPACK_EMBEDDING)?;
    Ok(())
}

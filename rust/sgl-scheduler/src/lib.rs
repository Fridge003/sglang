use pyo3::prelude::*;

mod check_finished;
mod decode_loop;
mod output_sender;
mod serialization;
mod stream_output;
mod types;
mod zmq_sender;

use check_finished::{batch_check_finished_rust, check_finished_rust, CheckFinishedResult};
use decode_loop::{process_decode_loop_rust, DecodeAction, DecodeReqInput};
use output_sender::RustOutputSender;

/// Python module for the sgl-scheduler Rust extension.
#[pymodule]
fn sgl_scheduler(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustOutputSender>()?;
    m.add_class::<CheckFinishedResult>()?;
    m.add_class::<DecodeReqInput>()?;
    m.add_class::<DecodeAction>()?;
    m.add_function(wrap_pyfunction!(check_finished_rust, m)?)?;
    m.add_function(wrap_pyfunction!(batch_check_finished_rust, m)?)?;
    m.add_function(wrap_pyfunction!(process_decode_loop_rust, m)?)?;
    // Export format prefix constants for Python-side dual-format receive
    m.add("FORMAT_PREFIX_PICKLE", types::FORMAT_PREFIX_PICKLE)?;
    m.add("FORMAT_PREFIX_MSGPACK_TOKEN", types::FORMAT_PREFIX_MSGPACK_TOKEN)?;
    m.add("FORMAT_PREFIX_MSGPACK_EMBEDDING", types::FORMAT_PREFIX_MSGPACK_EMBEDDING)?;
    Ok(())
}

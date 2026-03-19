use crate::types::{OutputMessage, FORMAT_PREFIX_MSGPACK_TOKEN};
use crossbeam_channel::Sender;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Serialize a StreamOutputData dict to msgpack bytes and enqueue for sending.
///
/// This replaces the Python-side path of:
///   1. Constructing BatchTokenIDOutput dataclass
///   2. Converting to dict
///   3. msgspec.msgpack.Encoder().encode(dict)
///   4. Passing bytes to Rust send_token_batch_bytes()
///
/// Instead, Python passes the pre-built dict directly and Rust serializes
/// via rmp-serde in one step, avoiding intermediate Python objects.
pub fn serialize_and_enqueue_token_output(
    tx: &Sender<OutputMessage>,
    data: &Bound<'_, PyDict>,
) -> PyResult<()> {
    // Convert the Python dict to an rmpv::Value, then serialize to msgpack.
    // We use rmpv::Value as an intermediate because the dict has heterogeneous
    // nested types (lists of lists, optional values, etc.).
    let value = pydict_to_rmpv(data)?;
    let bytes = rmp_serde::to_vec_named(&value).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("msgpack serialization error: {}", e))
    })?;

    // Prepend format prefix and enqueue as RawMessage (already prefixed)
    let mut msg = Vec::with_capacity(1 + bytes.len());
    msg.push(FORMAT_PREFIX_MSGPACK_TOKEN);
    msg.extend_from_slice(&bytes);

    tx.send(OutputMessage::RawMessage(msg))
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to enqueue token batch: {}",
                e
            ))
        })?;

    Ok(())
}

/// Convert a Python dict to rmpv::Value for msgpack serialization.
fn pydict_to_rmpv(dict: &Bound<'_, PyDict>) -> PyResult<rmpv::Value> {
    let mut map = Vec::new();
    for (key, value) in dict.iter() {
        let key_str: String = key.extract()?;
        let val = pyobj_to_rmpv(&value)?;
        map.push((rmpv::Value::String(key_str.into()), val));
    }
    Ok(rmpv::Value::Map(map))
}

/// Convert a Python object to rmpv::Value recursively.
fn pyobj_to_rmpv(obj: &Bound<'_, PyAny>) -> PyResult<rmpv::Value> {
    if obj.is_none() {
        return Ok(rmpv::Value::Nil);
    }

    // Check for bool BEFORE int (bool is a subclass of int in Python)
    if let Ok(val) = obj.extract::<bool>() {
        return Ok(rmpv::Value::Boolean(val));
    }

    if let Ok(val) = obj.extract::<i64>() {
        return Ok(rmpv::Value::Integer(val.into()));
    }

    if let Ok(val) = obj.extract::<f64>() {
        return Ok(rmpv::Value::F64(val));
    }

    if let Ok(val) = obj.extract::<String>() {
        return Ok(rmpv::Value::String(val.into()));
    }

    // Check for dict before list (both are iterable)
    if let Ok(dict) = obj.downcast::<PyDict>() {
        return pydict_to_rmpv(dict);
    }

    // List/tuple
    if let Ok(list) = obj.downcast::<pyo3::types::PyList>() {
        let mut arr = Vec::with_capacity(list.len());
        for item in list.iter() {
            arr.push(pyobj_to_rmpv(&item)?);
        }
        return Ok(rmpv::Value::Array(arr));
    }

    if let Ok(tuple) = obj.downcast::<pyo3::types::PyTuple>() {
        let mut arr = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            arr.push(pyobj_to_rmpv(&item)?);
        }
        return Ok(rmpv::Value::Array(arr));
    }

    // bytes
    if let Ok(val) = obj.extract::<Vec<u8>>() {
        return Ok(rmpv::Value::Binary(val));
    }

    // Fallback: try to convert via str representation
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "Cannot convert Python type {} to msgpack",
        obj.get_type().name()?
    )))
}

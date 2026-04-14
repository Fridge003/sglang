use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::mpsc::{self, Receiver, Sender};

use crate::tokenizer::RustTokenizer;

#[derive(Debug, Clone)]
pub enum ResponseChunk {
    Data(ResponseData),
    Finished(ResponseData),
    Error(String),
}

#[derive(Debug, Clone)]
pub struct ResponseData {
    pub text: Option<String>,
    pub output_ids: Option<Vec<i32>>,
    pub embedding: Option<Vec<f32>>,
    pub json_bytes: Option<Vec<u8>>,
    pub meta_info: HashMap<String, String>,
}

/// Holds a reference to the Python RuntimeHandle and manages per-request channels.
pub struct PyBridge {
    runtime_handle: PyObject,
    channels: Arc<Mutex<HashMap<String, Sender<ResponseChunk>>>>,
    rust_tokenizer: Option<RustTokenizer>,
    context_len: i32,
}

impl PyBridge {
    pub fn new(
        runtime_handle: PyObject,
        rust_tokenizer: Option<RustTokenizer>,
        context_len: i32,
    ) -> Self {
        Self {
            runtime_handle,
            channels: Arc::new(Mutex::new(HashMap::new())),
            rust_tokenizer,
            context_len,
        }
    }

    /// Access the Rust tokenizer (if available).
    pub fn rust_tokenizer(&self) -> Option<&RustTokenizer> {
        self.rust_tokenizer.as_ref()
    }

    /// Return the model's context length.
    pub fn context_len(&self) -> i32 {
        self.context_len
    }

    // ------------------------------------------------------------------
    // Channel + callback helpers
    // ------------------------------------------------------------------

    fn create_channel(&self, rid: &str) -> Receiver<ResponseChunk> {
        let (sender, receiver) = mpsc::channel(64);
        {
            let mut channels = self.channels.lock().unwrap();
            channels.insert(rid.to_string(), sender);
        }
        receiver
    }

    fn make_chunk_callback(
        &self,
        py: Python<'_>,
        rid: String,
        channels: Arc<Mutex<HashMap<String, Sender<ResponseChunk>>>>,
    ) -> PyResult<PyObject> {
        let callback = ChunkCallback { rid, channels };
        let py_callback = Py::new(py, callback)?;
        Ok(py_callback.into_any().into())
    }

    fn make_json_callback(
        &self,
        py: Python<'_>,
        rid: String,
        channels: Arc<Mutex<HashMap<String, Sender<ResponseChunk>>>>,
    ) -> PyResult<PyObject> {
        let callback = JsonChunkCallback { rid, channels };
        let py_callback = Py::new(py, callback)?;
        Ok(py_callback.into_any().into())
    }

    // ------------------------------------------------------------------
    // Consolidated request submission (generate / embed / classify)
    // ------------------------------------------------------------------

    /// Submit a generate or embed request by passing a pre-built dict to Python.
    ///
    /// `req_type` is "generate", "embed", or "classify".
    /// `req_dict` contains fields matching GenerateReqInput or EmbeddingReqInput.
    pub fn submit_request(
        &self,
        rid: &str,
        req_type: &str,
        req_dict: HashMap<String, serde_json::Value>,
    ) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid);
        let channels_ref = self.channels.clone();
        let rid_owned = rid.to_string();

        let result = Python::with_gil(|py| -> PyResult<()> {
            let py_req_dict = json_map_to_pydict(py, &req_dict)?;
            let callback = self.make_chunk_callback(py, rid_owned, channels_ref)?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("req_type", req_type)?;
            kwargs.set_item("req_dict", py_req_dict)?;
            kwargs.set_item("chunk_callback", callback)?;

            self.runtime_handle
                .call_method(py, "submit_request", (), Some(&kwargs))?;
            Ok(())
        });

        match result {
            Ok(()) => Ok(receiver),
            Err(err) => {
                self.remove_channel(rid);
                Err(err)
            }
        }
    }

    // ------------------------------------------------------------------
    // Abort
    // ------------------------------------------------------------------

    pub fn abort(&self, rid: &str) -> PyResult<()> {
        {
            let mut channels = self.channels.lock().unwrap();
            channels.remove(rid);
        }
        Python::with_gil(|py| {
            self.runtime_handle.call_method1(py, "abort", (rid,))?;
            Ok(())
        })
    }

    // ------------------------------------------------------------------
    // Info / control RPCs (synchronous, small data)
    // ------------------------------------------------------------------

    pub fn get_model_info(&self) -> PyResult<String> {
        Python::with_gil(|py| {
            let result = self.runtime_handle.call_method0(py, "get_model_info")?;
            result.extract::<String>(py)
        })
    }

    pub fn get_server_info(&self) -> PyResult<String> {
        Python::with_gil(|py| {
            let result = self.runtime_handle.call_method0(py, "get_server_info")?;
            result.extract::<String>(py)
        })
    }

    pub fn health_check(&self) -> PyResult<bool> {
        Python::with_gil(|py| {
            let result = self.runtime_handle.call_method0(py, "health_check")?;
            result.extract::<bool>(py)
        })
    }

    /// Tokenize via Python (fallback when Rust tokenizer unavailable).
    pub fn tokenize_py(&self, text: &str, add_special_tokens: bool) -> PyResult<String> {
        Python::with_gil(|py| {
            let result =
                self.runtime_handle
                    .call_method1(py, "tokenize", (text, add_special_tokens))?;
            result.extract::<String>(py)
        })
    }

    /// Detokenize via Python (fallback when Rust tokenizer unavailable).
    pub fn detokenize_py(&self, tokens: Vec<i32>) -> PyResult<String> {
        Python::with_gil(|py| {
            let result = self
                .runtime_handle
                .call_method1(py, "detokenize", (tokens,))?;
            result.extract::<String>(py)
        })
    }

    pub fn list_models(&self) -> PyResult<String> {
        Python::with_gil(|py| {
            let result = self.runtime_handle.call_method0(py, "list_models")?;
            result.extract::<String>(py)
        })
    }

    pub fn submit_get_load(&self, rid: &str) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid);
        let channels_ref = self.channels.clone();
        let rid_owned = rid.to_string();

        let result = Python::with_gil(|py| -> PyResult<()> {
            let callback = self.make_json_callback(py, rid_owned, channels_ref)?;
            self.runtime_handle
                .call_method1(py, "get_load", (callback,))?;
            Ok(())
        });

        match result {
            Ok(()) => Ok(receiver),
            Err(err) => {
                self.remove_channel(rid);
                Err(err)
            }
        }
    }

    pub fn submit_flush_cache(&self, rid: &str) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid);
        let channels_ref = self.channels.clone();
        let rid_owned = rid.to_string();

        let result = Python::with_gil(|py| -> PyResult<()> {
            let callback = self.make_json_callback(py, rid_owned, channels_ref)?;
            self.runtime_handle
                .call_method1(py, "flush_cache", (callback,))?;
            Ok(())
        });

        match result {
            Ok(()) => Ok(receiver),
            Err(err) => {
                self.remove_channel(rid);
                Err(err)
            }
        }
    }

    pub fn submit_pause_generation(
        &self,
        rid: &str,
        mode: &str,
    ) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid);
        let channels_ref = self.channels.clone();
        let rid_owned = rid.to_string();

        let result = Python::with_gil(|py| -> PyResult<()> {
            let callback = self.make_json_callback(py, rid_owned, channels_ref)?;
            self.runtime_handle
                .call_method1(py, "pause_generation", (mode, callback))?;
            Ok(())
        });

        match result {
            Ok(()) => Ok(receiver),
            Err(err) => {
                self.remove_channel(rid);
                Err(err)
            }
        }
    }

    pub fn submit_continue_generation(&self, rid: &str) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid);
        let channels_ref = self.channels.clone();
        let rid_owned = rid.to_string();

        let result = Python::with_gil(|py| -> PyResult<()> {
            let callback = self.make_json_callback(py, rid_owned, channels_ref)?;
            self.runtime_handle
                .call_method1(py, "continue_generation", (callback,))?;
            Ok(())
        });

        match result {
            Ok(()) => Ok(receiver),
            Err(err) => {
                self.remove_channel(rid);
                Err(err)
            }
        }
    }

    pub fn submit_start_profile(
        &self,
        rid: &str,
        output_dir: Option<&str>,
    ) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid);
        let channels_ref = self.channels.clone();
        let rid_owned = rid.to_string();

        let result = Python::with_gil(|py| -> PyResult<()> {
            let callback = self.make_json_callback(py, rid_owned, channels_ref)?;
            self.runtime_handle
                .call_method1(py, "start_profile", (output_dir, callback))?;
            Ok(())
        });

        match result {
            Ok(()) => Ok(receiver),
            Err(err) => {
                self.remove_channel(rid);
                Err(err)
            }
        }
    }

    pub fn submit_stop_profile(&self, rid: &str) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid);
        let channels_ref = self.channels.clone();
        let rid_owned = rid.to_string();

        let result = Python::with_gil(|py| -> PyResult<()> {
            let callback = self.make_json_callback(py, rid_owned, channels_ref)?;
            self.runtime_handle
                .call_method1(py, "stop_profile", (callback,))?;
            Ok(())
        });

        match result {
            Ok(()) => Ok(receiver),
            Err(err) => {
                self.remove_channel(rid);
                Err(err)
            }
        }
    }

    pub fn submit_update_weights(
        &self,
        rid: &str,
        model_path: &str,
        load_format: Option<&str>,
    ) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid);
        let channels_ref = self.channels.clone();
        let rid_owned = rid.to_string();

        let result = Python::with_gil(|py| -> PyResult<()> {
            let callback = self.make_json_callback(py, rid_owned, channels_ref)?;
            self.runtime_handle.call_method1(
                py,
                "update_weights_from_disk",
                (model_path, load_format, callback),
            )?;
            Ok(())
        });

        match result {
            Ok(()) => Ok(receiver),
            Err(err) => {
                self.remove_channel(rid);
                Err(err)
            }
        }
    }

    // ------------------------------------------------------------------
    // OpenAI pass-through RPCs
    // ------------------------------------------------------------------

    pub fn submit_openai(
        &self,
        rid: &str,
        method_name: &str,
        json_body: &[u8],
    ) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid);
        let channels_ref = self.channels.clone();
        let rid_owned = rid.to_string();

        let result = Python::with_gil(|py| -> PyResult<()> {
            let kwargs = PyDict::new(py);
            let py_bytes = PyBytes::new(py, json_body);
            kwargs.set_item("json_body", py_bytes)?;

            let callback = self.make_json_callback(py, rid_owned, channels_ref)?;
            kwargs.set_item("chunk_callback", callback)?;

            self.runtime_handle
                .call_method(py, method_name, (), Some(&kwargs))?;
            Ok(())
        });

        match result {
            Ok(()) => Ok(receiver),
            Err(err) => {
                self.remove_channel(rid);
                Err(err)
            }
        }
    }

    pub fn remove_channel(&self, rid: &str) {
        let mut channels = self.channels.lock().unwrap();
        channels.remove(rid);
    }
}

// ======================================================================
// Convert serde_json::Value map to PyDict
// ======================================================================

fn json_value_to_py<'py>(py: Python<'py>, v: &serde_json::Value) -> PyResult<PyObject> {
    match v {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.into_pyobject(py)?.to_owned().into_any().unbind()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_any().unbind())
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        serde_json::Value::Array(arr) => {
            let items: Vec<PyObject> = arr
                .iter()
                .map(|item| json_value_to_py(py, item))
                .collect::<PyResult<_>>()?;
            let py_list = PyList::new(py, &items)?;
            Ok(py_list.into_any().unbind())
        }
        serde_json::Value::Object(map) => {
            let py_dict = PyDict::new(py);
            for (k, val) in map {
                py_dict.set_item(k, json_value_to_py(py, val)?)?;
            }
            Ok(py_dict.into_any().unbind())
        }
    }
}

fn json_map_to_pydict<'py>(
    py: Python<'py>,
    map: &HashMap<String, serde_json::Value>,
) -> PyResult<Bound<'py, PyDict>> {
    let py_dict = PyDict::new(py);
    for (k, v) in map {
        py_dict.set_item(k, json_value_to_py(py, v)?)?;
    }
    Ok(py_dict)
}

fn try_send_chunk(
    rid: &str,
    channels: &Arc<Mutex<HashMap<String, Sender<ResponseChunk>>>>,
    sender: &Sender<ResponseChunk>,
    msg: ResponseChunk,
) -> PyResult<()> {
    match sender.try_send(msg) {
        Ok(()) => Ok(()),
        Err(TrySendError::Full(_)) => {
            let mut channels = channels.lock().unwrap();
            channels.remove(rid);
            Err(PyErr::new::<PyRuntimeError, _>(
                "gRPC response channel full: client not consuming",
            ))
        }
        Err(TrySendError::Closed(_)) => {
            let mut channels = channels.lock().unwrap();
            channels.remove(rid);
            Err(PyErr::new::<PyRuntimeError, _>("gRPC client disconnected"))
        }
    }
}

// ======================================================================
// Typed chunk callback (for SGLang-native RPCs: dict-based chunks)
// ======================================================================

#[pyclass]
struct ChunkCallback {
    rid: String,
    channels: Arc<Mutex<HashMap<String, Sender<ResponseChunk>>>>,
}

#[pymethods]
impl ChunkCallback {
    #[pyo3(signature = (chunk, finished=false, error=None))]
    fn __call__(
        &self,
        chunk: &Bound<'_, PyDict>,
        finished: bool,
        error: Option<String>,
    ) -> PyResult<()> {
        let channels = self.channels.lock().unwrap();
        let sender = match channels.get(&self.rid) {
            Some(s) => s.clone(),
            None => return Ok(()),
        };
        drop(channels);

        if let Some(err_msg) = error {
            try_send_chunk(
                &self.rid,
                &self.channels,
                &sender,
                ResponseChunk::Error(err_msg),
            )?;
            let mut channels = self.channels.lock().unwrap();
            channels.remove(&self.rid);
            return Ok(());
        }

        let text: Option<String> = chunk
            .get_item("text")?
            .and_then(|v| v.extract::<String>().ok());

        let output_ids: Option<Vec<i32>> = chunk
            .get_item("output_ids")?
            .and_then(|v| v.extract::<Vec<i32>>().ok());

        let embedding: Option<Vec<f32>> = chunk
            .get_item("embedding")?
            .and_then(|v| v.extract::<Vec<f32>>().ok());

        let meta_info = extract_meta_info(chunk);

        let data = ResponseData {
            text,
            output_ids,
            embedding,
            json_bytes: None,
            meta_info,
        };

        let msg = if finished {
            ResponseChunk::Finished(data)
        } else {
            ResponseChunk::Data(data)
        };

        try_send_chunk(&self.rid, &self.channels, &sender, msg)?;

        if finished {
            let mut channels = self.channels.lock().unwrap();
            channels.remove(&self.rid);
        }

        Ok(())
    }
}

// ======================================================================
// JSON chunk callback (for OpenAI pass-through RPCs: raw bytes)
// ======================================================================

#[pyclass]
struct JsonChunkCallback {
    rid: String,
    channels: Arc<Mutex<HashMap<String, Sender<ResponseChunk>>>>,
}

#[pymethods]
impl JsonChunkCallback {
    #[pyo3(signature = (chunk_bytes, finished=false, error=None))]
    fn __call__(
        &self,
        chunk_bytes: &Bound<'_, pyo3::PyAny>,
        finished: bool,
        error: Option<String>,
    ) -> PyResult<()> {
        let channels = self.channels.lock().unwrap();
        let sender = match channels.get(&self.rid) {
            Some(s) => s.clone(),
            None => return Ok(()),
        };
        drop(channels);

        if let Some(err_msg) = error {
            try_send_chunk(
                &self.rid,
                &self.channels,
                &sender,
                ResponseChunk::Error(err_msg),
            )?;
            let mut channels = self.channels.lock().unwrap();
            channels.remove(&self.rid);
            return Ok(());
        }

        let bytes_data: Vec<u8> = if let Ok(b) = chunk_bytes.extract::<Vec<u8>>() {
            b
        } else if let Ok(s) = chunk_bytes.extract::<String>() {
            s.into_bytes()
        } else {
            vec![]
        };

        let data = ResponseData {
            text: None,
            output_ids: None,
            embedding: None,
            json_bytes: Some(bytes_data),
            meta_info: HashMap::new(),
        };

        let msg = if finished {
            ResponseChunk::Finished(data)
        } else {
            ResponseChunk::Data(data)
        };

        try_send_chunk(&self.rid, &self.channels, &sender, msg)?;

        if finished {
            let mut channels = self.channels.lock().unwrap();
            channels.remove(&self.rid);
        }

        Ok(())
    }
}

fn extract_meta_info(chunk: &Bound<'_, PyDict>) -> HashMap<String, String> {
    let mut meta = HashMap::new();
    if let Ok(Some(meta_obj)) = chunk.get_item("meta_info") {
        if let Ok(meta_dict) = meta_obj.downcast::<PyDict>() {
            for (k, v) in meta_dict.iter() {
                if let (Ok(key), Ok(val)) = (k.extract::<String>(), v.str()) {
                    meta.insert(key, val.to_string());
                }
            }
        }
    }
    meta
}

use crossbeam_channel::{bounded, Receiver, Sender};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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

pub struct RequestChannel {
    pub receiver: Receiver<ResponseChunk>,
    sender: Sender<ResponseChunk>,
}

impl RequestChannel {
    fn new() -> Self {
        let (sender, receiver) = bounded(64);
        Self { receiver, sender }
    }
}

/// Holds a reference to the Python RuntimeHandle and manages per-request channels.
pub struct PyBridge {
    runtime_handle: PyObject,
    channels: Arc<Mutex<HashMap<String, Sender<ResponseChunk>>>>,
}

impl PyBridge {
    pub fn new(runtime_handle: PyObject) -> Self {
        Self {
            runtime_handle,
            channels: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    // ------------------------------------------------------------------
    // Channel + callback helpers
    // ------------------------------------------------------------------

    fn create_channel(&self, rid: &str) -> Receiver<ResponseChunk> {
        let chan = RequestChannel::new();
        let sender = chan.sender.clone();
        let receiver = chan.receiver.clone();
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

    fn set_trace_headers(
        &self,
        py: Python<'_>,
        kwargs: &Bound<'_, PyDict>,
        trace_headers: &Option<HashMap<String, String>>,
    ) -> PyResult<()> {
        if let Some(ref headers) = trace_headers {
            let py_headers = PyDict::new(py);
            for (k, v) in headers {
                py_headers.set_item(k, v)?;
            }
            kwargs.set_item("trace_headers", py_headers)?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Generate (text or tokenized)
    // ------------------------------------------------------------------

    pub fn submit_text_generate(
        &self,
        rid: &str,
        text: &str,
        sampling_params_json: &str,
        stream: bool,
        return_logprob: bool,
        top_logprobs_num: i32,
        logprob_start_len: i32,
        return_text_in_logprobs: bool,
        lora_path: Option<&str>,
        routing_key: Option<&str>,
        routed_dp_rank: Option<i32>,
        trace_headers: Option<HashMap<String, String>>,
    ) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid);
        let channels_ref = self.channels.clone();
        let rid_owned = rid.to_string();

        Python::with_gil(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("rid", rid)?;
            kwargs.set_item("text", text)?;
            kwargs.set_item("sampling_params_json", sampling_params_json)?;
            kwargs.set_item("stream", stream)?;
            kwargs.set_item("return_logprob", return_logprob)?;
            kwargs.set_item("top_logprobs_num", top_logprobs_num)?;
            kwargs.set_item("logprob_start_len", logprob_start_len)?;
            kwargs.set_item("return_text_in_logprobs", return_text_in_logprobs)?;
            if let Some(lp) = lora_path {
                kwargs.set_item("lora_path", lp)?;
            }
            if let Some(rk) = routing_key {
                kwargs.set_item("routing_key", rk)?;
            }
            if let Some(rank) = routed_dp_rank {
                kwargs.set_item("routed_dp_rank", rank)?;
            }
            self.set_trace_headers(py, &kwargs, &trace_headers)?;

            let callback = self.make_chunk_callback(py, rid_owned, channels_ref)?;
            kwargs.set_item("chunk_callback", callback)?;

            self.runtime_handle
                .call_method(py, "submit_generate", (), Some(&kwargs))?;
            Ok(receiver)
        })
    }

    pub fn submit_generate(
        &self,
        rid: &str,
        input_ids: Vec<i32>,
        sampling_params_json: &str,
        stream: bool,
        return_logprob: bool,
        top_logprobs_num: i32,
        logprob_start_len: i32,
        lora_path: Option<&str>,
        routing_key: Option<&str>,
        routed_dp_rank: Option<i32>,
        trace_headers: Option<HashMap<String, String>>,
    ) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid);
        let channels_ref = self.channels.clone();
        let rid_owned = rid.to_string();

        Python::with_gil(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("rid", rid)?;
            kwargs.set_item("input_ids", &input_ids)?;
            kwargs.set_item("sampling_params_json", sampling_params_json)?;
            kwargs.set_item("stream", stream)?;
            kwargs.set_item("return_logprob", return_logprob)?;
            kwargs.set_item("top_logprobs_num", top_logprobs_num)?;
            kwargs.set_item("logprob_start_len", logprob_start_len)?;
            if let Some(lp) = lora_path {
                kwargs.set_item("lora_path", lp)?;
            }
            if let Some(rk) = routing_key {
                kwargs.set_item("routing_key", rk)?;
            }
            if let Some(rank) = routed_dp_rank {
                kwargs.set_item("routed_dp_rank", rank)?;
            }
            self.set_trace_headers(py, &kwargs, &trace_headers)?;

            let callback = self.make_chunk_callback(py, rid_owned, channels_ref)?;
            kwargs.set_item("chunk_callback", callback)?;

            self.runtime_handle
                .call_method(py, "submit_generate", (), Some(&kwargs))?;
            Ok(receiver)
        })
    }

    // ------------------------------------------------------------------
    // Embed (text or tokenized)
    // ------------------------------------------------------------------

    pub fn submit_text_embed(
        &self,
        rid: &str,
        text: &str,
        routing_key: Option<&str>,
        trace_headers: Option<HashMap<String, String>>,
    ) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid);
        let channels_ref = self.channels.clone();
        let rid_owned = rid.to_string();

        Python::with_gil(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("rid", rid)?;
            kwargs.set_item("text", text)?;
            if let Some(rk) = routing_key {
                kwargs.set_item("routing_key", rk)?;
            }
            self.set_trace_headers(py, &kwargs, &trace_headers)?;

            let callback = self.make_chunk_callback(py, rid_owned, channels_ref)?;
            kwargs.set_item("chunk_callback", callback)?;

            self.runtime_handle
                .call_method(py, "submit_embed", (), Some(&kwargs))?;
            Ok(receiver)
        })
    }

    pub fn submit_embed(
        &self,
        rid: &str,
        input_ids: Vec<i32>,
        routing_key: Option<&str>,
        trace_headers: Option<HashMap<String, String>>,
    ) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid);
        let channels_ref = self.channels.clone();
        let rid_owned = rid.to_string();

        Python::with_gil(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("rid", rid)?;
            kwargs.set_item("input_ids", &input_ids)?;
            if let Some(rk) = routing_key {
                kwargs.set_item("routing_key", rk)?;
            }
            self.set_trace_headers(py, &kwargs, &trace_headers)?;

            let callback = self.make_chunk_callback(py, rid_owned, channels_ref)?;
            kwargs.set_item("chunk_callback", callback)?;

            self.runtime_handle
                .call_method(py, "submit_embed", (), Some(&kwargs))?;
            Ok(receiver)
        })
    }

    // ------------------------------------------------------------------
    // Classify (same path as embed)
    // ------------------------------------------------------------------

    pub fn submit_classify(
        &self,
        rid: &str,
        text: Option<&str>,
        input_ids: Option<Vec<i32>>,
        routing_key: Option<&str>,
        trace_headers: Option<HashMap<String, String>>,
    ) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid);
        let channels_ref = self.channels.clone();
        let rid_owned = rid.to_string();

        Python::with_gil(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("rid", rid)?;
            if let Some(t) = text {
                kwargs.set_item("text", t)?;
            }
            if let Some(ref ids) = input_ids {
                kwargs.set_item("input_ids", ids)?;
            }
            if let Some(rk) = routing_key {
                kwargs.set_item("routing_key", rk)?;
            }
            self.set_trace_headers(py, &kwargs, &trace_headers)?;

            let callback = self.make_chunk_callback(py, rid_owned, channels_ref)?;
            kwargs.set_item("chunk_callback", callback)?;

            self.runtime_handle
                .call_method(py, "submit_classify", (), Some(&kwargs))?;
            Ok(receiver)
        })
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

    pub fn tokenize(&self, text: &str, add_special_tokens: bool) -> PyResult<String> {
        Python::with_gil(|py| {
            let result = self
                .runtime_handle
                .call_method1(py, "tokenize", (text, add_special_tokens))?;
            result.extract::<String>(py)
        })
    }

    pub fn detokenize(&self, tokens: Vec<i32>) -> PyResult<String> {
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

        Python::with_gil(|py| {
            let callback = self.make_json_callback(py, rid_owned, channels_ref)?;
            self.runtime_handle
                .call_method1(py, "get_load", (callback,))?;
            Ok(receiver)
        })
    }

    pub fn submit_flush_cache(&self, rid: &str) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid);
        let channels_ref = self.channels.clone();
        let rid_owned = rid.to_string();

        Python::with_gil(|py| {
            let callback = self.make_json_callback(py, rid_owned, channels_ref)?;
            self.runtime_handle
                .call_method1(py, "flush_cache", (callback,))?;
            Ok(receiver)
        })
    }

    pub fn submit_pause_generation(
        &self,
        rid: &str,
        mode: &str,
    ) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid);
        let channels_ref = self.channels.clone();
        let rid_owned = rid.to_string();

        Python::with_gil(|py| {
            let callback = self.make_json_callback(py, rid_owned, channels_ref)?;
            self.runtime_handle
                .call_method1(py, "pause_generation", (mode, callback))?;
            Ok(receiver)
        })
    }

    pub fn submit_continue_generation(
        &self,
        rid: &str,
    ) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid);
        let channels_ref = self.channels.clone();
        let rid_owned = rid.to_string();

        Python::with_gil(|py| {
            let callback = self.make_json_callback(py, rid_owned, channels_ref)?;
            self.runtime_handle
                .call_method1(py, "continue_generation", (callback,))?;
            Ok(receiver)
        })
    }

    pub fn submit_start_profile(
        &self,
        rid: &str,
        output_dir: Option<&str>,
    ) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid);
        let channels_ref = self.channels.clone();
        let rid_owned = rid.to_string();

        Python::with_gil(|py| {
            let callback = self.make_json_callback(py, rid_owned, channels_ref)?;
            self.runtime_handle
                .call_method1(py, "start_profile", (output_dir, callback))?;
            Ok(receiver)
        })
    }

    pub fn submit_stop_profile(&self, rid: &str) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid);
        let channels_ref = self.channels.clone();
        let rid_owned = rid.to_string();

        Python::with_gil(|py| {
            let callback = self.make_json_callback(py, rid_owned, channels_ref)?;
            self.runtime_handle
                .call_method1(py, "stop_profile", (callback,))?;
            Ok(receiver)
        })
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

        Python::with_gil(|py| {
            let callback = self.make_json_callback(py, rid_owned, channels_ref)?;
            self.runtime_handle.call_method1(
                py,
                "update_weights_from_disk",
                (model_path, load_format, callback),
            )?;
            Ok(receiver)
        })
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

        Python::with_gil(|py| {
            let kwargs = PyDict::new(py);
            let py_bytes = PyBytes::new(py, json_body);
            kwargs.set_item("json_body", py_bytes)?;

            let callback = self.make_json_callback(py, rid_owned, channels_ref)?;
            kwargs.set_item("chunk_callback", callback)?;

            self.runtime_handle
                .call_method(py, method_name, (), Some(&kwargs))?;
            Ok(receiver)
        })
    }

    pub fn remove_channel(&self, rid: &str) {
        let mut channels = self.channels.lock().unwrap();
        channels.remove(rid);
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
            let _ = sender.send(ResponseChunk::Error(err_msg));
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

        let _ = sender.send(msg);

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
            let _ = sender.send(ResponseChunk::Error(err_msg));
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

        let _ = sender.send(msg);

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

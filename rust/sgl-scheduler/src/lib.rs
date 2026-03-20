use pyo3::ffi;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet};
use std::collections::HashSet;
use std::time::Instant;

// ============================================================================
// Result struct
// ============================================================================

#[pyclass(get_all)]
#[derive(Default)]
struct FastDecodeResult {
    /// finish_type: 1=to_finish, 2=length, 4=matched_token
    newly_finished_indices: Vec<usize>,
    finish_types: Vec<i32>,
    finish_matched_token_ids: Vec<i64>,
    grammar_indices: Vec<usize>,
    str_stop_check_indices: Vec<usize>,

    output_rids: Vec<Py<PyAny>>,
    output_http_worker_ipcs: Vec<Py<PyAny>>,
    output_finished_reasons: Vec<Py<PyAny>>,
    output_decoded_texts: Vec<Py<PyAny>>,
    output_decode_ids: Vec<Py<PyAny>>,
    output_read_offsets: Vec<i64>,
    output_ids: Vec<Py<PyAny>>,
    output_skip_special_tokens: Vec<bool>,
    output_spaces_between_special_tokens: Vec<bool>,
    output_no_stop_trim: Vec<bool>,
    output_prompt_tokens: Vec<i64>,
    output_completion_tokens: Vec<i64>,
    output_cached_tokens: Vec<i64>,
    output_cached_tokens_details: Vec<Py<PyAny>>,
    output_retraction_counts: Vec<i64>,
    output_time_stats: Vec<Py<PyAny>>,
    log_time_stats_indices: Vec<usize>,

    prof_cache_setup_us: f64,
    prof_loop1_us: f64,
    prof_loop2_us: f64,
}

#[pymethods]
impl FastDecodeResult {
    fn __repr__(&self) -> String {
        format!(
            "FastDecodeResult(finished={}, grammar={}, str_stop={}, outputs={})",
            self.newly_finished_indices.len(),
            self.grammar_indices.len(),
            self.str_stop_check_indices.len(),
            self.output_rids.len(),
        )
    }
}

// ============================================================================
// Fast dict-based attribute access via CPython C API
// ============================================================================

/// Get an object's __dict__ as a raw pointer via tp_dictoffset.
/// Falls back to PyObject_GetAttr("__dict__") if tp_dictoffset is not set.
/// The returned pointer is borrowed (no refcount change) when using tp_dictoffset.
#[inline(always)]
unsafe fn obj_dict(obj: *mut ffi::PyObject) -> *mut ffi::PyObject {
    let tp = ffi::Py_TYPE(obj);
    let offset = (*tp).tp_dictoffset;
    if offset > 0 {
        let dict_ptr = (obj as *const u8).offset(offset as isize) as *const *mut ffi::PyObject;
        let d = *dict_ptr;
        if !d.is_null() {
            return d;
        }
    }
    // Fallback: use the safe API
    let d = ffi::PyObject_GenericGetDict(obj, std::ptr::null_mut());
    if !d.is_null() {
        // GenericGetDict returns a new ref; decref to match borrowed semantics
        // This is safe because the dict is held alive by the object
        ffi::Py_DECREF(d);
    }
    d
}

/// Read a value from obj.__dict__[key]. Returns a borrowed reference (no refcount change).
/// key must be an interned string. Returns null if not found.
#[inline(always)]
unsafe fn dict_get(dict: *mut ffi::PyObject, key: *mut ffi::PyObject) -> *mut ffi::PyObject {
    // PyDict_GetItem returns a borrowed ref, does NOT set exception on missing key
    ffi::PyDict_GetItem(dict, key)
}

/// Set a value in obj.__dict__[key] = val.
#[inline(always)]
unsafe fn dict_set(dict: *mut ffi::PyObject, key: *mut ffi::PyObject, val: *mut ffi::PyObject) {
    ffi::PyDict_SetItem(dict, key, val);
}

/// Check if a Python object is None.
#[inline(always)]
unsafe fn is_none(obj: *mut ffi::PyObject) -> bool {
    obj == ffi::Py_None()
}

/// Check if a Python bool object is True.
#[inline(always)]
unsafe fn is_true(obj: *mut ffi::PyObject) -> bool {
    obj == ffi::Py_True()
}

// ============================================================================
// Helpers
// ============================================================================

fn py_set_to_hashset(obj: &Bound<'_, PyAny>) -> PyResult<HashSet<i64>> {
    let mut hs = HashSet::new();
    if obj.is_none() { return Ok(hs); }
    if let Ok(set) = obj.downcast::<PySet>() {
        for item in set.iter() { hs.insert(item.extract::<i64>()?); }
    } else {
        for item in obj.try_iter()? { hs.insert(item?.extract::<i64>()?); }
    }
    Ok(hs)
}

struct BatchSharedState {
    all_eos: HashSet<i64>,
    max_new_tokens: i64,
    ignore_eos: bool,
    has_stop_strs: bool,
}

impl BatchSharedState {
    fn from_first_active_req(
        py: Python<'_>, reqs: &Bound<'_, PyList>,
    ) -> PyResult<Self> {
        for i in 0..reqs.len() {
            let req = reqs.get_item(i)?;
            if !req.getattr(intern!(py, "finished_reason"))?.is_none() { continue; }

            let mut all_eos = HashSet::new();
            let tokenizer = req.getattr(intern!(py, "tokenizer"))?;
            if !tokenizer.is_none() {
                all_eos.insert(tokenizer.getattr(intern!(py, "eos_token_id"))?.extract()?);
                let additional = tokenizer.getattr(intern!(py, "additional_stop_token_ids"))?;
                if !additional.is_none() {
                    for id in &py_set_to_hashset(&additional)? { all_eos.insert(*id); }
                }
            }
            let sp = req.getattr(intern!(py, "sampling_params"))?;
            let max_new_tokens: i64 = sp.getattr(intern!(py, "max_new_tokens"))?.extract()?;
            let ignore_eos: bool = sp.getattr(intern!(py, "ignore_eos"))?.extract()?;
            let stop_ids = sp.getattr(intern!(py, "stop_token_ids"))?;
            if !stop_ids.is_none() {
                for id in &py_set_to_hashset(&stop_ids)? { all_eos.insert(*id); }
            }
            let has_stop_strs = sp.getattr(intern!(py, "stop_strs"))?.len()? > 0
                || sp.getattr(intern!(py, "stop_regex_strs"))?.len()? > 0;
            return Ok(Self { all_eos, max_new_tokens, ignore_eos, has_stop_strs });
        }
        Ok(Self { all_eos: HashSet::new(), max_new_tokens: i64::MAX, ignore_eos: false, has_stop_strs: false })
    }
}

// ============================================================================
// Main function
// ============================================================================

/// Rust-accelerated decode output processing.
///
/// Single Rust call replaces both the Python pre-loop and the finish checking.
/// Uses direct CPython C API (PyDict_GetItem/SetItem, PyList_Append) to bypass
/// PyO3's safe wrapper overhead for the hot inner loop.
#[pyfunction]
#[pyo3(signature = (
    reqs, next_token_ids,
    is_multimodal_gen, stream_interval, default_force_stream_interval,
    enable_request_time_stats_logging, get_cached_tokens_details_fn,
))]
fn process_batch_result_decode_fast(
    py: Python<'_>,
    reqs: &Bound<'_, PyList>,
    next_token_ids: Vec<i64>,
    is_multimodal_gen: bool,
    stream_interval: i32,
    default_force_stream_interval: i32,
    enable_request_time_stats_logging: bool,
    get_cached_tokens_details_fn: &Bound<'_, PyAny>,
) -> PyResult<FastDecodeResult> {
    let mut result = FastDecodeResult::default();
    let n = reqs.len();

    let t_start = Instant::now();
    let shared = BatchSharedState::from_first_active_req(py, reqs)?;

    // Pre-intern all attribute name strings for dict_get
    let k_finished_reason = intern!(py, "finished_reason");
    let k_is_retracted = intern!(py, "is_retracted");
    let k_output_ids = intern!(py, "output_ids");
    let k_time_stats = intern!(py, "time_stats");
    let k_last_decode_finish_time = intern!(py, "last_decode_finish_time");
    let k_to_finish = intern!(py, "to_finish");
    let k_grammar = intern!(py, "grammar");

    // Single timestamp for all reqs
    let time_mod = py.import(intern!(py, "time"))?;
    let batch_ts = time_mod.call_method0(intern!(py, "perf_counter"))?;
    let batch_ts_ptr = batch_ts.as_ptr();

    result.prof_cache_setup_us = t_start.elapsed().as_secs_f64() * 1e6;
    let t_loop1 = Instant::now();

    // ========================================================================
    // Main loop: append token + finish check using direct C API
    // ========================================================================
    // Uses PyDict_GetItem (borrowed ref, no refcount overhead) instead of
    // PyObject_GetAttr (full descriptor protocol). ~2x faster per access.
    let mut output_ids_lens: Vec<i64> = vec![0; n];

    for i in 0..n {
        // Direct list item access (borrowed ref, no bounds check)
        let req_ptr = unsafe { ffi::PyList_GET_ITEM(reqs.as_ptr(), i as ffi::Py_ssize_t) };
        let req_dict = unsafe { obj_dict(req_ptr) };
        if req_dict.is_null() {
            // Fallback for objects without __dict__ (shouldn't happen with Req)
            continue;
        }

        // Skip finished/retracted via direct dict lookup
        let fr = unsafe { dict_get(req_dict, k_finished_reason.as_ptr()) };
        if !fr.is_null() && !unsafe { is_none(fr) } {
            continue;
        }
        let ir = unsafe { dict_get(req_dict, k_is_retracted.as_ptr()) };
        if !ir.is_null() && unsafe { is_true(ir) } {
            continue;
        }

        // Append token via PyList_Append (C API, no method lookup)
        let oids = unsafe { dict_get(req_dict, k_output_ids.as_ptr()) };
        if oids.is_null() { continue; }
        let token_obj = next_token_ids[i].into_pyobject(py)?;
        unsafe { ffi::PyList_Append(oids, token_obj.as_ptr()); }
        let olen = unsafe { ffi::PyList_GET_SIZE(oids) } as i64;
        output_ids_lens[i] = olen;

        // Set timestamp via direct dict access on time_stats
        let ts_obj = unsafe { dict_get(req_dict, k_time_stats.as_ptr()) };
        if !ts_obj.is_null() {
            let ts_dict = unsafe { obj_dict(ts_obj) };
            if !ts_dict.is_null() {
                unsafe { dict_set(ts_dict, k_last_decode_finish_time.as_ptr(), batch_ts_ptr); }
            }
        }

        // --- Finish checks ---
        let next_token_id = next_token_ids[i];

        // Length (pure Rust)
        if olen >= shared.max_new_tokens {
            result.newly_finished_indices.push(i);
            result.finish_types.push(2);
            result.finish_matched_token_ids.push(0);
            continue;
        }

        // EOS token (pure Rust)
        if !shared.ignore_eos && shared.all_eos.contains(&next_token_id) {
            result.newly_finished_indices.push(i);
            result.finish_types.push(4);
            result.finish_matched_token_ids.push(next_token_id);
            continue;
        }

        // to_finish (direct dict lookup, no descriptor protocol)
        let tf = unsafe { dict_get(req_dict, k_to_finish.as_ptr()) };
        if !tf.is_null() && !unsafe { is_none(tf) } {
            result.newly_finished_indices.push(i);
            result.finish_types.push(1);
            result.finish_matched_token_ids.push(0);
            continue;
        }

        // Grammar (direct dict lookup)
        let gr = unsafe { dict_get(req_dict, k_grammar.as_ptr()) };
        if !gr.is_null() && !unsafe { is_none(gr) } {
            result.grammar_indices.push(i);
            continue;
        }

        if shared.has_stop_strs {
            result.str_stop_check_indices.push(i);
        }
    }

    result.prof_loop1_us = t_loop1.elapsed().as_secs_f64() * 1e6;
    let t_loop2 = Instant::now();

    // ========================================================================
    // Apply finish reasons (only for finished reqs — small count)
    // ========================================================================
    if !result.newly_finished_indices.is_empty() {
        let sb = py.import(intern!(py, "sglang.srt.managers.schedule_batch"))?;
        let cls_length = sb.getattr(intern!(py, "FINISH_LENGTH"))?;
        let cls_token = sb.getattr(intern!(py, "FINISH_MATCHED_TOKEN"))?;
        let cls_str = sb.getattr(intern!(py, "FINISH_MATCHED_STR"))?;

        for (j, &idx) in result.newly_finished_indices.iter().enumerate() {
            let req = reqs.get_item(idx)?;
            match result.finish_types[j] {
                1 => {
                    let tf = req.getattr(intern!(py, "to_finish"))?;
                    req.setattr(intern!(py, "finished_reason"), &tf)?;
                    req.setattr(intern!(py, "to_finish"), py.None())?;
                }
                2 => {
                    let sp = req.getattr(intern!(py, "sampling_params"))?;
                    let mnt: i64 = sp.getattr(intern!(py, "max_new_tokens"))?.extract()?;
                    req.setattr(intern!(py, "finished_reason"), cls_length.call1((mnt,))?)?;
                    req.setattr(intern!(py, "finished_len"), mnt)?;
                }
                4 => {
                    let tid = result.finish_matched_token_ids[j];
                    req.setattr(intern!(py, "finished_reason"), cls_token.call1((tid,))?)?;
                    req.setattr(intern!(py, "finished_len"), output_ids_lens[idx])?;
                }
                5 => {
                    req.setattr(intern!(py, "finished_reason"), cls_str.call1(("NaN happened",))?)?;
                }
                _ => {}
            }
            let ts = req.getattr(intern!(py, "time_stats"))?;
            let ct = time_mod.call_method0(intern!(py, "perf_counter"))?;
            ts.setattr(intern!(py, "completion_time"), &ct)?;
            ts.getattr(intern!(py, "trace_ctx"))?.call_method0(intern!(py, "abort"))?;
        }
    }

    // ========================================================================
    // Stream output collection
    // ========================================================================
    for i in 0..n {
        let req = reqs.get_item(i)?;

        if is_multimodal_gen {
            let tf = req.getattr(intern!(py, "to_finish"))?;
            if !tf.is_none() { continue; }
        }

        let fr = req.getattr(intern!(py, "finished_reason"))?;
        let is_fin = !fr.is_none();

        let should_output = if is_fin {
            let fin_out = req.getattr(intern!(py, "finished_output"))?;
            if !fin_out.is_none() {
                if enable_request_time_stats_logging { result.log_time_stats_indices.push(i); }
                continue;
            }
            req.setattr(intern!(py, "finished_output"), true)?;
            let fl = req.getattr(intern!(py, "finished_len"))?;
            if fl.is_none() { req.setattr(intern!(py, "finished_len"), output_ids_lens[i])?; }
            true
        } else {
            let olen = output_ids_lens[i] as i32;
            if olen == 0 { continue; }
            let stream: bool = req.getattr(intern!(py, "stream"))?.extract()?;
            if stream {
                let sp = req.getattr(intern!(py, "sampling_params"))?;
                let rsi = sp.getattr(intern!(py, "stream_interval"))?;
                let eff = if rsi.is_none() { stream_interval } else { rsi.extract::<i32>()? };
                let base = if !is_multimodal_gen && eff > 1 { olen % eff == 1 } else { olen % eff == 0 };
                if base {
                    let stop: bool = req.call_method0(intern!(py, "check_match_stop_str_prefix"))?.extract()?;
                    !stop
                } else { false }
            } else if !is_multimodal_gen {
                olen % default_force_stream_interval == 0
            } else { false }
        };

        if should_output {
            let sto: i64 = req.getattr(intern!(py, "send_token_offset"))?.extract()?;
            result.output_rids.push(req.getattr(intern!(py, "rid"))?.unbind());
            result.output_http_worker_ipcs.push(req.getattr(intern!(py, "http_worker_ipc"))?.unbind());
            result.output_finished_reasons.push(if is_fin {
                fr.call_method0(intern!(py, "to_json"))?.unbind()
            } else { py.None() });
            result.output_decoded_texts.push(req.getattr(intern!(py, "decoded_text"))?.unbind());

            let dt = req.call_method0(intern!(py, "init_incremental_detokenize"))?;
            let dids = dt.get_item(0)?;
            let ro: i64 = dt.get_item(1)?.extract()?;
            let dlen = dids.len()? as i64;
            if is_multimodal_gen {
                result.output_decode_ids.push(dids.unbind());
            } else {
                let sdo: i64 = req.getattr(intern!(py, "send_decode_id_offset"))?.extract()?;
                result.output_decode_ids.push(
                    dids.get_item(pyo3::types::PySlice::new(py, sdo as isize, dlen as isize, 1))?.unbind());
            }
            req.setattr(intern!(py, "send_decode_id_offset"), dlen)?;
            result.output_read_offsets.push(ro);

            let ots = req.getattr(intern!(py, "output_ids_through_stop"))?;
            let ots_len = ots.len()? as i64;
            result.output_ids.push(
                ots.get_item(pyo3::types::PySlice::new(py, sto as isize, ots_len as isize, 1))?.unbind());
            req.setattr(intern!(py, "send_token_offset"), ots_len)?;

            let sp = req.getattr(intern!(py, "sampling_params"))?;
            result.output_skip_special_tokens.push(sp.getattr(intern!(py, "skip_special_tokens"))?.extract()?);
            result.output_spaces_between_special_tokens.push(sp.getattr(intern!(py, "spaces_between_special_tokens"))?.extract()?);
            result.output_no_stop_trim.push(sp.getattr(intern!(py, "no_stop_trim"))?.extract()?);
            result.output_prompt_tokens.push(req.getattr(intern!(py, "origin_input_ids"))?.len()? as i64);
            result.output_completion_tokens.push(ots_len);
            result.output_cached_tokens.push(req.getattr(intern!(py, "cached_tokens"))?.extract()?);
            result.output_cached_tokens_details.push(get_cached_tokens_details_fn.call1((&req,))?.unbind());
            result.output_retraction_counts.push(req.getattr(intern!(py, "retraction_count"))?.extract()?);
            result.output_time_stats.push(req.getattr(intern!(py, "time_stats"))?.unbind());
        }
        if is_fin && enable_request_time_stats_logging { result.log_time_stats_indices.push(i); }
    }

    result.prof_loop2_us = t_loop2.elapsed().as_secs_f64() * 1e6;
    Ok(result)
}

#[pymodule]
fn sgl_scheduler_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FastDecodeResult>()?;
    m.add_function(wrap_pyfunction!(process_batch_result_decode_fast, m)?)?;
    Ok(())
}

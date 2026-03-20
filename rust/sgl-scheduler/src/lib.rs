use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyList, PySet};
use std::collections::HashSet;
use std::time::Instant;

// ============================================================================
// Result struct returned to Python
// ============================================================================

#[pyclass(get_all)]
#[derive(Default)]
struct FastDecodeResult {
    /// Indices of reqs that finished, with their finish type and matched token.
    /// finish_type: 1=to_finish, 2=length, 4=matched_token, 5=vocab_boundary
    newly_finished_indices: Vec<usize>,
    finish_types: Vec<i32>,
    finish_matched_token_ids: Vec<i64>,

    /// Indices of reqs with grammar (need Python accept_token)
    grammar_indices: Vec<usize>,
    /// Indices of reqs needing str-based finish check
    str_stop_check_indices: Vec<usize>,

    // Stream output fields for BatchTokenIDOutput
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

    // Profiling (microseconds)
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
// Helpers
// ============================================================================

fn py_set_to_hashset(obj: &Bound<'_, PyAny>) -> PyResult<HashSet<i64>> {
    let mut hs = HashSet::new();
    if obj.is_none() {
        return Ok(hs);
    }
    if let Ok(set) = obj.downcast::<PySet>() {
        for item in set.iter() {
            hs.insert(item.extract::<i64>()?);
        }
    } else {
        for item in obj.try_iter()? {
            hs.insert(item?.extract::<i64>()?);
        }
    }
    Ok(hs)
}

/// Batch-wide shared state cached from the first active req.
struct BatchSharedState {
    all_eos: HashSet<i64>,
    max_new_tokens: i64,
    ignore_eos: bool,
    has_stop_strs: bool,
}

impl BatchSharedState {
    fn from_first_active_req(
        py: Python<'_>,
        reqs: &Bound<'_, PyList>,
        output_ids_lens: &[i64],
    ) -> PyResult<Self> {
        for i in 0..reqs.len() {
            if output_ids_lens[i] == 0 {
                continue; // skipped (finished/retracted)
            }
            let req = reqs.get_item(i)?;

            // Tokenizer EOS
            let mut all_eos = HashSet::new();
            let tokenizer = req.getattr(intern!(py, "tokenizer"))?;
            if !tokenizer.is_none() {
                all_eos.insert(tokenizer.getattr(intern!(py, "eos_token_id"))?.extract()?);
                let additional = tokenizer.getattr(intern!(py, "additional_stop_token_ids"))?;
                if !additional.is_none() {
                    for id in &py_set_to_hashset(&additional)? {
                        all_eos.insert(*id);
                    }
                }
            }

            // Sampling params
            let sp = req.getattr(intern!(py, "sampling_params"))?;
            let max_new_tokens: i64 = sp.getattr(intern!(py, "max_new_tokens"))?.extract()?;
            let ignore_eos: bool = sp.getattr(intern!(py, "ignore_eos"))?.extract()?;
            let stop_ids = sp.getattr(intern!(py, "stop_token_ids"))?;
            if !stop_ids.is_none() {
                for id in &py_set_to_hashset(&stop_ids)? {
                    all_eos.insert(*id);
                }
            }
            let has_stop_strs = sp.getattr(intern!(py, "stop_strs"))?.len()? > 0
                || sp.getattr(intern!(py, "stop_regex_strs"))?.len()? > 0;

            return Ok(Self { all_eos, max_new_tokens, ignore_eos, has_stop_strs });
        }
        Ok(Self {
            all_eos: HashSet::new(),
            max_new_tokens: i64::MAX,
            ignore_eos: false,
            has_stop_strs: false,
        })
    }
}

// ============================================================================
// Main function
// ============================================================================

/// Rust-accelerated decode output processing.
///
/// Python pre-loop has already:
///   - Appended next_token_ids[i] to each active req.output_ids
///   - Set req.time_stats.last_decode_finish_time
///   - Built output_ids_lens (0 for skipped reqs)
///
/// This function handles:
///   1. Finish checking (length, EOS, to_finish, grammar detection)
///   2. Applying finish reasons to Python objects
///   3. Stream output data collection
#[pyfunction]
#[pyo3(signature = (
    reqs, next_token_ids, output_ids_lens,
    is_multimodal_gen, stream_interval, default_force_stream_interval,
    enable_request_time_stats_logging, get_cached_tokens_details_fn,
))]
fn process_batch_result_decode_fast(
    py: Python<'_>,
    reqs: &Bound<'_, PyList>,
    next_token_ids: Vec<i64>,
    output_ids_lens: Vec<i64>,
    is_multimodal_gen: bool,
    stream_interval: i32,
    default_force_stream_interval: i32,
    enable_request_time_stats_logging: bool,
    get_cached_tokens_details_fn: &Bound<'_, PyAny>,
) -> PyResult<FastDecodeResult> {
    let mut result = FastDecodeResult::default();
    let n = reqs.len();

    let t_start = Instant::now();
    let shared = BatchSharedState::from_first_active_req(py, reqs, &output_ids_lens)?;
    let time_mod = py.import(intern!(py, "time"))?;
    result.prof_cache_setup_us = t_start.elapsed().as_secs_f64() * 1e6;

    let t_loop1 = Instant::now();

    // ========================================================================
    // Finish checking — single pass over reqs
    // ========================================================================
    // Token append and timestamp already done in Python.
    // Per-req: 2 getattrs (to_finish, grammar) for common case (both None).
    // Fast-path length/EOS checks are pure Rust (0 Python calls).
    for i in 0..n {
        let olen = output_ids_lens[i];
        if olen == 0 {
            continue; // skipped (finished/retracted)
        }
        let next_token_id = next_token_ids[i];

        // Fast-path: length check (pure Rust)
        if olen >= shared.max_new_tokens {
            result.newly_finished_indices.push(i);
            result.finish_types.push(2);
            result.finish_matched_token_ids.push(0);
            continue;
        }

        // Fast-path: EOS token check (pure Rust)
        if !shared.ignore_eos && shared.all_eos.contains(&next_token_id) {
            result.newly_finished_indices.push(i);
            result.finish_types.push(4);
            result.finish_matched_token_ids.push(next_token_id);
            continue;
        }

        // Slow-path: per-req Python checks (2 getattrs, almost always None)
        let req = reqs.get_item(i)?;

        let to_finish = req.getattr(intern!(py, "to_finish"))?;
        if !to_finish.is_none() {
            result.newly_finished_indices.push(i);
            result.finish_types.push(1);
            result.finish_matched_token_ids.push(0);
            continue;
        }

        let grammar = req.getattr(intern!(py, "grammar"))?;
        if !grammar.is_none() {
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
            if olen == 0 { continue; } // skipped

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

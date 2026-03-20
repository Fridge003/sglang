use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyList, PySet};
use std::collections::HashSet;

/// Result of the fast decode processing loop.
/// Contains indices for Python fallback and pre-collected stream output data.
#[pyclass(get_all)]
#[derive(Default)]
struct FastDecodeResult {
    // Check-finish results — indices for Python fallback
    newly_finished_indices: Vec<usize>,
    grammar_indices: Vec<usize>,
    str_stop_check_indices: Vec<usize>,

    // Stream output — pre-collected lists for BatchTokenIDOutput
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

/// Helper to extract a Python set of ints into a HashSet<i64>.
fn py_set_to_hashset(obj: &Bound<'_, PyAny>) -> PyResult<HashSet<i64>> {
    let mut result = HashSet::new();
    if obj.is_none() {
        return Ok(result);
    }
    if let Ok(set) = obj.downcast::<PySet>() {
        for item in set.iter() {
            result.insert(item.extract::<i64>()?);
        }
    } else {
        let iter = obj.try_iter()?;
        for item in iter {
            let item: Bound<'_, PyAny> = item?;
            result.insert(item.extract::<i64>()?);
        }
    }
    Ok(result)
}

/// Cached tokenizer fields that are the same for all reqs in a batch.
struct CachedTokenizer {
    has_tokenizer: bool,
    eos_token_id: i64,
    additional_stop_token_ids: HashSet<i64>,
}

impl CachedTokenizer {
    /// Read tokenizer fields once from the first req that has a non-None tokenizer.
    fn from_reqs(py: Python<'_>, reqs: &Bound<'_, PyList>) -> PyResult<Self> {
        for i in 0..reqs.len() {
            let req = reqs.get_item(i)?;
            let tokenizer = req.getattr(intern!(py, "tokenizer"))?;
            if !tokenizer.is_none() {
                let eos_token_id: i64 =
                    tokenizer.getattr(intern!(py, "eos_token_id"))?.extract()?;
                let additional_obj =
                    tokenizer.getattr(intern!(py, "additional_stop_token_ids"))?;
                let additional_stop_token_ids = py_set_to_hashset(&additional_obj)?;
                return Ok(CachedTokenizer {
                    has_tokenizer: true,
                    eos_token_id,
                    additional_stop_token_ids,
                });
            }
        }
        Ok(CachedTokenizer {
            has_tokenizer: false,
            eos_token_id: -1,
            additional_stop_token_ids: HashSet::new(),
        })
    }

    fn matches(&self, token_id: i64) -> bool {
        if !self.has_tokenizer {
            return false;
        }
        token_id == self.eos_token_id || self.additional_stop_token_ids.contains(&token_id)
    }
}

/// Fast Rust implementation of the check-finish + stream-output loops
/// in process_batch_result_decode.
#[pyfunction]
#[pyo3(signature = (
    reqs,
    next_token_ids,
    enable_overlap,
    is_multimodal_gen,
    stream_interval,
    default_force_stream_interval,
    enable_request_time_stats_logging,
    disagg_decode_offload,
    get_cached_tokens_details_fn,
))]
fn process_batch_result_decode_fast(
    py: Python<'_>,
    reqs: &Bound<'_, PyList>,
    next_token_ids: Vec<i64>,
    enable_overlap: bool,
    is_multimodal_gen: bool,
    stream_interval: i32,
    default_force_stream_interval: i32,
    enable_request_time_stats_logging: bool,
    disagg_decode_offload: bool,
    get_cached_tokens_details_fn: &Bound<'_, PyAny>,
) -> PyResult<FastDecodeResult> {
    let mut result = FastDecodeResult::default();
    let n = reqs.len();
    let _ = disagg_decode_offload; // handled in Python fallback

    // === Pre-loop caching ===

    // Cache tokenizer fields (same for all reqs)
    let cached_tok = CachedTokenizer::from_reqs(py, reqs)?;

    // Cache schedule_batch module import (used only on finish, but import is cheap when cached)
    let schedule_batch = py.import(intern!(py, "sglang.srt.managers.schedule_batch"))?;
    let finish_length_cls = schedule_batch.getattr(intern!(py, "FINISH_LENGTH"))?;
    let finish_matched_token_cls = schedule_batch.getattr(intern!(py, "FINISH_MATCHED_TOKEN"))?;
    let finish_matched_str_cls = schedule_batch.getattr(intern!(py, "FINISH_MATCHED_STR"))?;

    // Take a single timestamp for all set_last_decode_finish_time calls in this batch.
    // The Python method just does: if ts is None: ts = time.perf_counter(); self.last_decode_finish_time = ts
    // We do the perf_counter() once and set the attribute directly.
    let time_mod = py.import(intern!(py, "time"))?;
    let batch_decode_ts = time_mod.call_method0(intern!(py, "perf_counter"))?;

    // ========================
    // LOOP 1: Check-finish
    // ========================
    for i in 0..n {
        let req = reqs.get_item(i)?;
        let next_token_id = next_token_ids[i];

        // Skip finished/retracted in overlap mode
        if enable_overlap {
            let finished_reason = req.getattr(intern!(py, "finished_reason"))?;
            if !finished_reason.is_none() {
                continue;
            }
            let is_retracted: bool = req.getattr(intern!(py, "is_retracted"))?.extract()?;
            if is_retracted {
                continue;
            }
        }

        // Append token: req.output_ids.append(next_token_id)
        let output_ids = req.getattr(intern!(py, "output_ids"))?;
        output_ids.call_method1(intern!(py, "append"), (next_token_id,))?;

        // Inline set_last_decode_finish_time: directly set the attribute with cached timestamp
        let time_stats = req.getattr(intern!(py, "time_stats"))?;
        time_stats.setattr(intern!(py, "last_decode_finish_time"), &batch_decode_ts)?;

        // === Inline check_finished ===
        let finished_reason = req.getattr(intern!(py, "finished_reason"))?;
        if !finished_reason.is_none() {
            continue;
        }

        let mut newly_finished = false;

        // Check to_finish
        let to_finish = req.getattr(intern!(py, "to_finish"))?;
        if !to_finish.is_none() {
            req.setattr(intern!(py, "finished_reason"), &to_finish)?;
            req.setattr(intern!(py, "to_finish"), py.None())?;
            newly_finished = true;
        }

        if !newly_finished {
            let output_ids_len: i64 = output_ids.len()? as i64;
            let sampling_params = req.getattr(intern!(py, "sampling_params"))?;
            let max_new_tokens: i64 =
                sampling_params.getattr(intern!(py, "max_new_tokens"))?.extract()?;

            if output_ids_len >= max_new_tokens {
                let reason = finish_length_cls.call1((max_new_tokens,))?;
                req.setattr(intern!(py, "finished_reason"), &reason)?;
                req.setattr(intern!(py, "finished_len"), max_new_tokens)?;
                newly_finished = true;
            }

            if !newly_finished {
                let grammar = req.getattr(intern!(py, "grammar"))?;
                if !grammar.is_none() {
                    let is_terminated: bool =
                        grammar.call_method0(intern!(py, "is_terminated"))?.extract()?;
                    if is_terminated {
                        let last_token = output_ids.get_item(output_ids.len()? - 1)?;
                        let reason = finish_matched_token_cls.call1((last_token,))?;
                        req.setattr(intern!(py, "finished_reason"), &reason)?;
                        newly_finished = true;
                    } else {
                        result.grammar_indices.push(i);
                    }
                }

                if !newly_finished {
                    let ignore_eos: bool =
                        sampling_params.getattr(intern!(py, "ignore_eos"))?.extract()?;

                    if !ignore_eos {
                        let mut matched_eos = false;

                        // Check stop_token_ids (per-req sampling param)
                        let stop_token_ids_obj =
                            sampling_params.getattr(intern!(py, "stop_token_ids"))?;
                        if !stop_token_ids_obj.is_none() {
                            let stop_set = py_set_to_hashset(&stop_token_ids_obj)?;
                            if !stop_set.is_empty() {
                                matched_eos |= stop_set.contains(&next_token_id);
                            }
                        }

                        // Check eos_token_ids (per-req override)
                        if !matched_eos {
                            let eos_token_ids_obj =
                                req.getattr(intern!(py, "eos_token_ids"))?;
                            if !eos_token_ids_obj.is_none() {
                                let eos_set = py_set_to_hashset(&eos_token_ids_obj)?;
                                matched_eos |= eos_set.contains(&next_token_id);
                            }
                        }

                        // Check tokenizer eos (cached — no per-req Python call)
                        if !matched_eos {
                            matched_eos |= cached_tok.matches(next_token_id);
                        }

                        if matched_eos {
                            let reason =
                                finish_matched_token_cls.call1((next_token_id,))?;
                            req.setattr(intern!(py, "finished_reason"), &reason)?;
                            let olen = output_ids.len()? as i64;
                            req.setattr(intern!(py, "finished_len"), olen)?;
                            newly_finished = true;
                        }
                    }

                    if !newly_finished {
                        // Check vocab boundary
                        let vocab_size: i64 =
                            req.getattr(intern!(py, "vocab_size"))?.extract()?;
                        if next_token_id > vocab_size || next_token_id < 0 {
                            let reason =
                                finish_matched_str_cls.call1(("NaN happened",))?;
                            req.setattr(intern!(py, "finished_reason"), &reason)?;
                            newly_finished = true;
                        }
                    }

                    if !newly_finished {
                        // Check stop_strs - need Python fallback
                        let stop_strs = sampling_params.getattr(intern!(py, "stop_strs"))?;
                        let stop_strs_len: usize = stop_strs.len()?;
                        let stop_regex_strs =
                            sampling_params.getattr(intern!(py, "stop_regex_strs"))?;
                        let stop_regex_len: usize = stop_regex_strs.len()?;
                        if stop_strs_len > 0 || stop_regex_len > 0 {
                            result.str_stop_check_indices.push(i);
                        }
                    }
                }
            }
        }

        if newly_finished {
            result.newly_finished_indices.push(i);
            // Inline set_completion_time: set attribute + call trace_ctx.abort()
            let completion_ts = time_mod.call_method0(intern!(py, "perf_counter"))?;
            time_stats.setattr(intern!(py, "completion_time"), &completion_ts)?;
            let trace_ctx = time_stats.getattr(intern!(py, "trace_ctx"))?;
            trace_ctx.call_method0(intern!(py, "abort"))?;
        }
    }

    // ========================
    // LOOP 2: Stream output
    // ========================
    for i in 0..n {
        let req = reqs.get_item(i)?;

        // Skip multimodal gen aborted reqs
        if is_multimodal_gen {
            let to_finish = req.getattr(intern!(py, "to_finish"))?;
            if !to_finish.is_none() {
                continue;
            }
        }

        let finished_reason = req.getattr(intern!(py, "finished_reason"))?;
        let is_finished = !finished_reason.is_none();
        let should_output: bool;

        if is_finished {
            let finished_output = req.getattr(intern!(py, "finished_output"))?;
            if !finished_output.is_none() {
                if enable_request_time_stats_logging {
                    result.log_time_stats_indices.push(i);
                }
                continue;
            }
            req.setattr(intern!(py, "finished_output"), true)?;
            let finished_len = req.getattr(intern!(py, "finished_len"))?;
            if finished_len.is_none() {
                let output_ids = req.getattr(intern!(py, "output_ids"))?;
                let olen = output_ids.len()? as i64;
                req.setattr(intern!(py, "finished_len"), olen)?;
            }
            should_output = true;
        } else {
            let stream_flag: bool = req.getattr(intern!(py, "stream"))?.extract()?;
            let output_ids = req.getattr(intern!(py, "output_ids"))?;
            let output_len = output_ids.len()? as i32;

            if stream_flag {
                let sampling_params = req.getattr(intern!(py, "sampling_params"))?;
                let req_stream_interval =
                    sampling_params.getattr(intern!(py, "stream_interval"))?;
                let effective_interval = if req_stream_interval.is_none() {
                    stream_interval
                } else {
                    req_stream_interval.extract::<i32>()?
                };

                let base_should = if !is_multimodal_gen && effective_interval > 1 {
                    output_len % effective_interval == 1
                } else {
                    output_len % effective_interval == 0
                };

                if base_should {
                    let stop_match: bool = req
                        .call_method0(intern!(py, "check_match_stop_str_prefix"))?
                        .extract()?;
                    should_output = !stop_match;
                } else {
                    should_output = false;
                }
            } else {
                should_output = if !is_multimodal_gen {
                    output_len % default_force_stream_interval == 0
                } else {
                    false
                };
            }
        }

        if should_output {
            let send_token_offset: i64 =
                req.getattr(intern!(py, "send_token_offset"))?.extract()?;

            // rid
            let rid = req.getattr(intern!(py, "rid"))?;
            result.output_rids.push(rid.unbind());

            // http_worker_ipc
            let ipc = req.getattr(intern!(py, "http_worker_ipc"))?;
            result.output_http_worker_ipcs.push(ipc.unbind());

            // finished_reason.to_json() or None
            if is_finished {
                let json = finished_reason.call_method0(intern!(py, "to_json"))?;
                result.output_finished_reasons.push(json.unbind());
            } else {
                result.output_finished_reasons.push(py.None());
            }

            // decoded_text
            let decoded_text = req.getattr(intern!(py, "decoded_text"))?;
            result.output_decoded_texts.push(decoded_text.unbind());

            // init_incremental_detokenize -> (decode_ids, read_offset)
            let detok_result =
                req.call_method0(intern!(py, "init_incremental_detokenize"))?;
            let decode_ids_full = detok_result.get_item(0)?;
            let read_offset: i64 = detok_result.get_item(1)?.extract()?;

            let decode_ids_full_len = decode_ids_full.len()? as i64;

            if is_multimodal_gen {
                result.output_decode_ids.push(decode_ids_full.unbind());
            } else {
                let send_decode_id_offset: i64 =
                    req.getattr(intern!(py, "send_decode_id_offset"))?.extract()?;
                let sliced = decode_ids_full.get_item(pyo3::types::PySlice::new(
                    py,
                    send_decode_id_offset as isize,
                    decode_ids_full_len as isize,
                    1,
                ))?;
                result.output_decode_ids.push(sliced.unbind());
            }

            req.setattr(intern!(py, "send_decode_id_offset"), decode_ids_full_len)?;
            result.output_read_offsets.push(read_offset);

            // output_ids_through_stop[send_token_offset:]
            let output_ids_through_stop =
                req.getattr(intern!(py, "output_ids_through_stop"))?;
            let output_ids_through_stop_len = output_ids_through_stop.len()? as i64;
            let sliced_output =
                output_ids_through_stop.get_item(pyo3::types::PySlice::new(
                    py,
                    send_token_offset as isize,
                    output_ids_through_stop_len as isize,
                    1,
                ))?;
            result.output_ids.push(sliced_output.unbind());
            req.setattr(
                intern!(py, "send_token_offset"),
                output_ids_through_stop_len,
            )?;

            // sampling_params fields
            let sampling_params = req.getattr(intern!(py, "sampling_params"))?;
            let skip_special: bool = sampling_params
                .getattr(intern!(py, "skip_special_tokens"))?
                .extract()?;
            let spaces_between: bool = sampling_params
                .getattr(intern!(py, "spaces_between_special_tokens"))?
                .extract()?;
            let no_stop_trim: bool = sampling_params
                .getattr(intern!(py, "no_stop_trim"))?
                .extract()?;
            result.output_skip_special_tokens.push(skip_special);
            result
                .output_spaces_between_special_tokens
                .push(spaces_between);
            result.output_no_stop_trim.push(no_stop_trim);

            // prompt_tokens = len(origin_input_ids)
            let origin_input_ids = req.getattr(intern!(py, "origin_input_ids"))?;
            let prompt_tokens = origin_input_ids.len()? as i64;
            result.output_prompt_tokens.push(prompt_tokens);

            // completion_tokens
            result
                .output_completion_tokens
                .push(output_ids_through_stop_len);

            // cached_tokens
            let cached_tokens: i64 =
                req.getattr(intern!(py, "cached_tokens"))?.extract()?;
            result.output_cached_tokens.push(cached_tokens);

            // cached_tokens_details via callback
            let details = get_cached_tokens_details_fn.call1((&req,))?;
            result.output_cached_tokens_details.push(details.unbind());

            // retraction_count
            let retraction_count: i64 =
                req.getattr(intern!(py, "retraction_count"))?.extract()?;
            result.output_retraction_counts.push(retraction_count);

            // time_stats
            let time_stats = req.getattr(intern!(py, "time_stats"))?;
            result.output_time_stats.push(time_stats.unbind());
        }

        // Time stats logging
        if is_finished && enable_request_time_stats_logging {
            result.log_time_stats_indices.push(i);
        }
    }

    Ok(result)
}

#[pymodule]
fn sgl_scheduler_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FastDecodeResult>()?;
    m.add_function(wrap_pyfunction!(process_batch_result_decode_fast, m)?)?;
    Ok(())
}

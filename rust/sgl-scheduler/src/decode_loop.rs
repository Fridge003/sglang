use crate::check_finished::check_finished_rust;
use pyo3::prelude::*;

/// Action that Python must execute after the Rust decode loop.
#[pyclass]
#[derive(Clone, Debug)]
pub struct DecodeAction {
    /// Index in the batch (i.e., position in batch.reqs)
    #[pyo3(get)]
    pub req_idx: usize,

    /// Action type:
    ///   "skip"           — req was skipped (overlap + finished/retracted)
    ///   "finished"       — req finished, Python should release KV cache etc.
    ///   "not_finished"   — req not finished, no special action needed
    ///   "grammar_accept" — Python should call grammar.accept_token
    ///   "grammar_error"  — grammar accept failed (Rust can't detect this, placeholder)
    #[pyo3(get)]
    pub action: String,

    /// The new_accepted_len for this request
    #[pyo3(get)]
    pub new_accepted_len: usize,

    /// The finish check result (reused from Phase 3)
    /// finish_type: "none" | "length" | "matched_token" | "matched_str" | "vocab_boundary"
    #[pyo3(get)]
    pub finish_type: String,
    #[pyo3(get)]
    pub finish_match_int: i64,
    #[pyo3(get)]
    pub finish_match_str: String,
    #[pyo3(get)]
    pub finish_finished_len: i64,
    #[pyo3(get)]
    pub finish_modified_offset: i64,
    #[pyo3(get)]
    pub finish_modified_value: i64,
}

/// Per-request input data for the decode loop.
/// Python pre-extracts these from Req objects to avoid repeated Python↔Rust crossings.
#[pyclass]
#[derive(Clone)]
pub struct DecodeReqInput {
    #[pyo3(get, set)]
    pub already_finished: bool,
    #[pyo3(get, set)]
    pub is_retracted: bool,
    #[pyo3(get, set)]
    pub output_ids_len: usize,
    #[pyo3(get, set)]
    pub max_new_tokens: usize,
    #[pyo3(get, set)]
    pub ignore_eos: bool,
    #[pyo3(get, set)]
    pub stop_token_ids: Vec<i64>,
    #[pyo3(get, set)]
    pub vocab_size: i64,
    #[pyo3(get, set)]
    pub first_stop_token_id: i64,
    #[pyo3(get, set)]
    pub stop_strs: Vec<String>,
    #[pyo3(get, set)]
    pub tail_str: String,
    #[pyo3(get, set)]
    pub decoded_text: String,
    #[pyo3(get, set)]
    pub has_grammar: bool,
    #[pyo3(get, set)]
    pub has_to_finish: bool,
    #[pyo3(get, set)]
    pub has_stop_regex: bool,
    #[pyo3(get, set)]
    pub return_logprob: bool,
    #[pyo3(get, set)]
    pub top_logprobs_num: i32,
    #[pyo3(get, set)]
    pub has_token_ids_logprob: bool,
}

#[pymethods]
impl DecodeReqInput {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        already_finished: bool,
        is_retracted: bool,
        output_ids_len: usize,
        max_new_tokens: usize,
        ignore_eos: bool,
        stop_token_ids: Vec<i64>,
        vocab_size: i64,
        first_stop_token_id: i64,
        stop_strs: Vec<String>,
        tail_str: String,
        decoded_text: String,
        has_grammar: bool,
        has_to_finish: bool,
        has_stop_regex: bool,
        return_logprob: bool,
        top_logprobs_num: i32,
        has_token_ids_logprob: bool,
    ) -> Self {
        Self {
            already_finished,
            is_retracted,
            output_ids_len,
            max_new_tokens,
            ignore_eos,
            stop_token_ids,
            vocab_size,
            first_stop_token_id,
            stop_strs,
            tail_str,
            decoded_text,
            has_grammar,
            has_to_finish,
            has_stop_regex,
            return_logprob,
            top_logprobs_num,
            has_token_ids_logprob,
        }
    }
}

/// Process the decode loop for a batch of requests.
///
/// This is the Rust equivalent of the inner for-loop in process_batch_result_decode.
/// It handles:
/// - Skipping finished/retracted requests (overlap mode)
/// - output_ids.append (via new_accepted_len tracking)
/// - check_finished (via Phase 3 Rust implementation)
/// - Logprob value/idx appending decisions
///
/// Returns a list of DecodeAction, one per request, telling Python what to do.
///
/// Arguments:
///   req_inputs: Per-request pre-extracted data
///   next_token_ids: The next token IDs from the model (flat list, one per req for non-spec)
///   enable_overlap: Whether overlap scheduling is enabled
///   is_spec_none: batch.spec_algorithm.is_none()
///   is_spec_v2: batch.is_spec_v2
///   return_logprob: batch.return_logprob
///   next_token_logprobs: Optional logprob values (one per req, or nested for spec_v2)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn process_decode_loop_rust(
    req_inputs: Vec<DecodeReqInput>,
    next_token_ids_flat: Vec<Vec<i64>>,
    enable_overlap: bool,
    is_spec_none: bool,
    is_spec_v2: bool,
    _return_logprob: bool,
) -> Vec<DecodeAction> {
    let n = req_inputs.len();
    let mut actions = Vec::with_capacity(n);

    for i in 0..n {
        let inp = &req_inputs[i];
        let next_tokens = &next_token_ids_flat[i];

        // Skip if overlap mode and already finished/retracted
        if enable_overlap && (inp.already_finished || inp.is_retracted) {
            actions.push(DecodeAction {
                req_idx: i,
                action: "skip".into(),
                new_accepted_len: 0,
                finish_type: "none".into(),
                finish_match_int: 0,
                finish_match_str: String::new(),
                finish_finished_len: -1,
                finish_modified_offset: -1,
                finish_modified_value: 0,
            });
            continue;
        }

        // Determine new_accepted_len
        let new_accepted_len = if is_spec_none {
            1
        } else if is_spec_v2 {
            next_tokens.len()
        } else {
            1
        };

        // New output_ids_len after appending
        let new_output_ids_len = inp.output_ids_len + new_accepted_len;

        // Run check_finished
        let new_accepted_tokens = next_tokens.clone();
        let cf_result = check_finished_rust(
            new_output_ids_len,
            inp.max_new_tokens,
            new_accepted_tokens,
            inp.ignore_eos,
            inp.stop_token_ids.clone(),
            inp.vocab_size,
            inp.first_stop_token_id,
            inp.stop_strs.clone(),
            inp.tail_str.clone(),
            inp.decoded_text.clone(),
            inp.has_grammar,
            inp.has_to_finish,
            inp.has_stop_regex,
        );

        let is_finished = cf_result.finish_type != "none";

        let action_str = if is_finished {
            "finished"
        } else {
            "not_finished"
        };

        actions.push(DecodeAction {
            req_idx: i,
            action: action_str.into(),
            new_accepted_len,
            finish_type: cf_result.finish_type,
            finish_match_int: cf_result.match_int,
            finish_match_str: cf_result.match_str,
            finish_finished_len: cf_result.finished_len,
            finish_modified_offset: cf_result.modified_output_id_offset,
            finish_modified_value: cf_result.modified_output_id_value,
        });
    }

    actions
}

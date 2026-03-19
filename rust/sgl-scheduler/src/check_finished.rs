use pyo3::prelude::*;

/// Result of check_finished for a single request.
/// Python uses these fields to set the appropriate finished_reason on the Req.
#[pyclass]
#[derive(Clone, Debug)]
pub struct CheckFinishedResult {
    /// "none" | "length" | "matched_token" | "matched_str" | "matched_regex" | "vocab_boundary"
    #[pyo3(get)]
    pub finish_type: String,

    /// For "matched_token": the token id. For "length": max_new_tokens.
    /// For "matched_str"/"matched_regex": unused (matched string is in match_str).
    /// For "vocab_boundary": the offset position.
    #[pyo3(get)]
    pub match_int: i64,

    /// For "matched_str"/"matched_regex": the matched string.
    #[pyo3(get)]
    pub match_str: String,

    /// For "length" and "matched_token" and "vocab_boundary": the finished_len.
    /// -1 means not set (Python should not update finished_len).
    #[pyo3(get)]
    pub finished_len: i64,

    /// If true, the output_ids at the given offset was modified (vocab boundary case).
    #[pyo3(get)]
    pub modified_output_id_offset: i64,
    #[pyo3(get)]
    pub modified_output_id_value: i64,
}

impl CheckFinishedResult {
    fn none() -> Self {
        Self {
            finish_type: "none".into(),
            match_int: 0,
            match_str: String::new(),
            finished_len: -1,
            modified_output_id_offset: -1,
            modified_output_id_value: 0,
        }
    }

    fn length(max_new_tokens: i64) -> Self {
        Self {
            finish_type: "length".into(),
            match_int: max_new_tokens,
            match_str: String::new(),
            finished_len: max_new_tokens,
            modified_output_id_offset: -1,
            modified_output_id_value: 0,
        }
    }

    fn matched_token(token_id: i64, finished_len: i64) -> Self {
        Self {
            finish_type: "matched_token".into(),
            match_int: token_id,
            match_str: String::new(),
            finished_len,
            modified_output_id_offset: -1,
            modified_output_id_value: 0,
        }
    }

    fn matched_str(s: String) -> Self {
        Self {
            finish_type: "matched_str".into(),
            match_int: 0,
            match_str: s,
            finished_len: -1,
            modified_output_id_offset: -1,
            modified_output_id_value: 0,
        }
    }

    fn matched_regex(s: String) -> Self {
        Self {
            finish_type: "matched_regex".into(),
            match_int: 0,
            match_str: s,
            finished_len: -1,
            modified_output_id_offset: -1,
            modified_output_id_value: 0,
        }
    }

    fn vocab_boundary(offset: i64, replacement_token: i64, finished_len: i64) -> Self {
        Self {
            finish_type: "vocab_boundary".into(),
            match_int: 0,
            match_str: "NaN happened".into(),
            finished_len,
            modified_output_id_offset: offset,
            modified_output_id_value: replacement_token,
        }
    }
}

/// Check finished conditions for a single request, in Rust.
///
/// This re-implements the logic from Req.check_finished() except:
/// - Grammar check (grammar.is_terminated) — caller checks has_grammar flag
/// - Stop regex (requires Python regex) — handled via tail_str from Python
/// - Tokenizer-based stop tokens — passed in as stop_token_ids
///
/// Arguments:
///   output_ids_len: len(req.output_ids)
///   max_new_tokens: req.sampling_params.max_new_tokens
///   new_accepted_tokens: req.output_ids[-new_accepted_len:]
///   ignore_eos: req.sampling_params.ignore_eos
///   stop_token_ids: combined set of all stop token IDs (EOS + custom)
///   vocab_size: req.vocab_size
///   first_stop_token_id: fallback token for vocab boundary replacement
///   stop_strs: req.sampling_params.stop_strs
///   tail_str: pre-computed tail string for stop string checking
///   decoded_text: req.decoded_text
///   has_grammar: whether grammar is not None (caller checks separately)
///   has_to_finish: whether req.to_finish is set (caller handles)
///   has_stop_regex: whether stop_regex_strs is non-empty (caller checks with Python regex)
#[pyfunction]
pub fn check_finished_rust(
    output_ids_len: usize,
    max_new_tokens: usize,
    new_accepted_tokens: Vec<i64>,
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
) -> CheckFinishedResult {
    // Already finished or to_finish — caller handles these before calling Rust
    if has_to_finish {
        return CheckFinishedResult::none();
    }

    // Check max_new_tokens
    if output_ids_len >= max_new_tokens {
        return CheckFinishedResult::length(max_new_tokens as i64);
    }

    // Grammar check — return "none" so caller can check Python-side
    if has_grammar {
        // Caller must check grammar.is_terminated() in Python
        return CheckFinishedResult::none();
    }

    // Check token-based finish
    if !ignore_eos {
        for (i, &token_id) in new_accepted_tokens.iter().enumerate() {
            let matched = stop_token_ids.contains(&token_id);
            if matched {
                let matched_pos = output_ids_len - new_accepted_tokens.len() + i;
                return CheckFinishedResult::matched_token(token_id, (matched_pos + 1) as i64);
            }
        }
    }

    // Check vocab boundary
    for (i, &token_id) in new_accepted_tokens.iter().enumerate() {
        if token_id > vocab_size || token_id < 0 {
            let offset = (output_ids_len - new_accepted_tokens.len() + i) as i64;
            return CheckFinishedResult::vocab_boundary(
                offset,
                first_stop_token_id,
                offset + 1,
            );
        }
    }

    // Check stop strings
    if !stop_strs.is_empty() && !tail_str.is_empty() {
        for stop_str in &stop_strs {
            if tail_str.contains(stop_str.as_str()) || decoded_text.contains(stop_str.as_str()) {
                return CheckFinishedResult::matched_str(stop_str.clone());
            }
        }
    }

    // Stop regex — caller handles in Python if has_stop_regex
    // We return "none" here so the caller can do the regex check
    if has_stop_regex {
        return CheckFinishedResult::none();
    }

    CheckFinishedResult::none()
}

/// Batch version: check finished for multiple requests at once.
/// Returns a Vec of CheckFinishedResult, one per request.
///
/// Each request's data is provided as parallel arrays (one element per request).
#[pyfunction]
pub fn batch_check_finished_rust(
    output_ids_lens: Vec<usize>,
    max_new_tokens_list: Vec<usize>,
    new_accepted_tokens_list: Vec<Vec<i64>>,
    ignore_eos_list: Vec<bool>,
    stop_token_ids_list: Vec<Vec<i64>>,
    vocab_sizes: Vec<i64>,
    first_stop_token_ids: Vec<i64>,
    stop_strs_list: Vec<Vec<String>>,
    tail_strs: Vec<String>,
    decoded_texts: Vec<String>,
    has_grammar_list: Vec<bool>,
    has_to_finish_list: Vec<bool>,
    has_stop_regex_list: Vec<bool>,
) -> Vec<CheckFinishedResult> {
    let n = output_ids_lens.len();
    let mut results = Vec::with_capacity(n);

    for i in 0..n {
        results.push(check_finished_rust(
            output_ids_lens[i],
            max_new_tokens_list[i],
            new_accepted_tokens_list[i].clone(),
            ignore_eos_list[i],
            stop_token_ids_list[i].clone(),
            vocab_sizes[i],
            first_stop_token_ids[i],
            stop_strs_list[i].clone(),
            tail_strs[i].clone(),
            decoded_texts[i].clone(),
            has_grammar_list[i],
            has_to_finish_list[i],
            has_stop_regex_list[i],
        ));
    }

    results
}

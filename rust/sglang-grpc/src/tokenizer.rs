use std::path::Path;
use tokenizers::Tokenizer;

/// Rust-native tokenizer wrapping HuggingFace's `tokenizers` crate.
///
/// This is the same Rust library that Python's `tokenizers` package wraps,
/// so it produces identical output for the same tokenizer.json file.
pub struct RustTokenizer {
    inner: Tokenizer,
    context_len: i32,
}

impl RustTokenizer {
    /// Load a tokenizer from a model directory (looks for `tokenizer.json`).
    /// Returns `None` if the file doesn't exist or can't be loaded.
    pub fn from_model_path(model_path: &str, context_len: i32) -> Option<Self> {
        let tokenizer_json = Path::new(model_path).join("tokenizer.json");
        if !tokenizer_json.exists() {
            tracing::info!(
                "No tokenizer.json found at {:?}, Rust tokenizer disabled",
                tokenizer_json
            );
            return None;
        }

        match Tokenizer::from_file(&tokenizer_json) {
            Ok(inner) => {
                tracing::info!(
                    "Rust tokenizer loaded from {:?} (context_len={})",
                    tokenizer_json,
                    context_len
                );
                Some(Self { inner, context_len })
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to load Rust tokenizer from {:?}: {}. Falling back to Python.",
                    tokenizer_json,
                    e
                );
                None
            }
        }
    }

    /// Tokenize text, returning token IDs.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, String> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| format!("Tokenization failed: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| format!("Detokenization failed: {}", e))
    }

    /// Return the model's context length.
    pub fn context_len(&self) -> i32 {
        self.context_len
    }
}

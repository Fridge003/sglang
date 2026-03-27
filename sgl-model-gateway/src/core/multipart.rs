//! Lightweight multipart/form-data field extraction.
//!
//! This is intentionally minimal: we only need to extract the `model` text field
//! from diffusion endpoint payloads (`/v1/videos`, `/v1/images/edits`).
//! We do NOT re-parse or re-encode the body — the original bytes are forwarded
//! to the worker unchanged.

use bytes::Bytes;

/// Extract the `model` field value from a `multipart/form-data` body.
///
/// Returns `None` if the `Content-Type` has no boundary, the body cannot be
/// scanned, or no `model` field is present.
///
/// # How it works
/// We scan the raw bytes for the multipart boundary, then for each part we check
/// whether the `Content-Disposition` header names the field `"model"`. When found
/// we return the trimmed UTF-8 text value. Binary parts (files) are skipped by
/// looking only at the headers section (before the blank line) to determine name,
/// avoiding any attempt to decode binary data.
pub fn extract_model_from_multipart(body: &Bytes, content_type: &str) -> Option<String> {
    let boundary = parse_boundary(content_type)?;

    // Delimiter bytes: "--{boundary}"
    let delimiter = format!("--{}", boundary);
    let delim_bytes = delimiter.as_bytes();

    // Closing delimiter: "--{boundary}--"
    let closing = format!("--{}--", boundary);
    let _closing_bytes = closing.as_bytes();

    let data = body.as_ref();
    let mut pos = 0;

    while pos < data.len() {
        // Find next delimiter
        let delim_pos = find_bytes(data, delim_bytes, pos)?;

        // If this is the closing delimiter, we're done
        let after_delim = delim_pos + delim_bytes.len();
        if data.get(after_delim..after_delim + 2) == Some(b"--") {
            break;
        }
        // Skip the CRLF after the delimiter
        let part_start = skip_crlf(data, after_delim);

        // Find the blank line (CRLF CRLF or LF LF) that separates headers from body
        let (headers_end, body_start) = find_headers_end(data, part_start)?;

        let headers_bytes = &data[part_start..headers_end];

        // Check if Content-Disposition names this field "model"
        if is_model_field(headers_bytes) {
            // Find end of this part: next delimiter
            let next_delim = find_bytes(data, delim_bytes, body_start)
                .unwrap_or(data.len());
            // Trim trailing CRLF before the delimiter
            let value_end = trim_end_crlf(data, body_start, next_delim);
            let value = std::str::from_utf8(&data[body_start..value_end]).ok()?;
            return Some(value.trim().to_string());
        }

        // Advance past this part to find the next delimiter
        pos = after_delim;
    }

    None
}

/// Parse the `boundary` parameter from a `Content-Type: multipart/form-data; boundary=...` value.
fn parse_boundary(content_type: &str) -> Option<&str> {
    // Find "boundary=" (case-insensitive per RFC)
    let lower = content_type.to_lowercase();
    let idx = lower.find("boundary=")?;
    let rest = &content_type[idx + "boundary=".len()..];
    // Boundary may be quoted: boundary="abc" or unquoted: boundary=abc
    let boundary = if rest.starts_with('"') {
        rest.trim_start_matches('"')
            .split('"')
            .next()?
    } else {
        rest.split(|c: char| c == ';' || c.is_whitespace())
            .next()?
    };
    if boundary.is_empty() { None } else { Some(boundary) }
}

/// Find the first occurrence of `needle` in `haystack` starting at `from`.
fn find_bytes(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || from >= haystack.len() {
        return None;
    }
    let search = &haystack[from..];
    // Use memchr for the first byte, then verify the full needle
    let first = needle[0];
    let mut offset = 0;
    while offset + needle.len() <= search.len() {
        if let Some(rel) = memchr::memchr(first, &search[offset..]) {
            let abs = from + offset + rel;
            if haystack.get(abs..abs + needle.len()) == Some(needle) {
                return Some(abs);
            }
            offset += rel + 1;
        } else {
            break;
        }
    }
    None
}

/// Skip a leading CRLF or LF at position `pos`.
fn skip_crlf(data: &[u8], pos: usize) -> usize {
    match data.get(pos..pos + 2) {
        Some(b"\r\n") => pos + 2,
        _ => match data.get(pos) {
            Some(b'\n') => pos + 1,
            _ => pos,
        },
    }
}

/// Find the blank line separating headers from body.
/// Returns `(headers_end, body_start)` where `headers_end` is the index just
/// before the blank line and `body_start` is just after it.
fn find_headers_end(data: &[u8], from: usize) -> Option<(usize, usize)> {
    // Look for CRLFCRLF or LFLF
    if let Some(pos) = find_bytes(data, b"\r\n\r\n", from) {
        return Some((pos, pos + 4));
    }
    if let Some(pos) = find_bytes(data, b"\n\n", from) {
        return Some((pos, pos + 2));
    }
    None
}

/// Return true if the part headers declare `name="model"` in Content-Disposition.
fn is_model_field(headers: &[u8]) -> bool {
    // Search for 'name="model"' (case-insensitive header name, case-sensitive value)
    let haystack = std::str::from_utf8(headers).unwrap_or("");
    // Header value matching: look for the Content-Disposition line
    for line in haystack.lines() {
        let lower_line = line.to_lowercase();
        if lower_line.starts_with("content-disposition:") {
            // Check for name="model" or name='model'
            if contains_field_name(line, "model") {
                return true;
            }
        }
    }
    false
}

/// Check whether a Content-Disposition header value contains `name="<field_name>"`.
fn contains_field_name(header_value: &str, field_name: &str) -> bool {
    // Handles: name="model", name='model', name=model
    let lower = header_value.to_lowercase();
    let target_quoted = format!("name=\"{}\"", field_name);
    let target_unquoted = format!("name={}", field_name);
    lower.contains(&target_quoted) || lower.contains(&target_unquoted)
}

/// Trim trailing CRLF or LF before `end` (the delimiter position).
fn trim_end_crlf(data: &[u8], _start: usize, end: usize) -> usize {
    if end >= 2 && data.get(end - 2..end) == Some(b"\r\n") {
        end - 2
    } else if end >= 1 && data.get(end - 1..end) == Some(b"\n") {
        end - 1
    } else {
        end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_multipart(boundary: &str, fields: &[(&str, &str)]) -> (String, Bytes) {
        let content_type = format!("multipart/form-data; boundary={}", boundary);
        let mut body = String::new();
        for (name, value) in fields {
            body.push_str(&format!("--{}\r\n", boundary));
            body.push_str(&format!(
                "Content-Disposition: form-data; name=\"{}\"\r\n\r\n",
                name
            ));
            body.push_str(value);
            body.push_str("\r\n");
        }
        body.push_str(&format!("--{}--\r\n", boundary));
        (content_type, Bytes::from(body))
    }

    #[test]
    fn extracts_model_field() {
        let (ct, body) = make_multipart(
            "boundary123",
            &[
                ("prompt", "a raccoon in a forest"),
                ("model", "Wan-AI/Wan2.2-T2V-A14B-Diffusers"),
                ("height", "720"),
            ],
        );
        assert_eq!(
            extract_model_from_multipart(&body, &ct),
            Some("Wan-AI/Wan2.2-T2V-A14B-Diffusers".to_string())
        );
    }

    #[test]
    fn model_field_first() {
        let (ct, body) = make_multipart(
            "abc",
            &[
                ("model", "/storage/models/zai-org/GLM-5-FP8"),
                ("prompt", "hello"),
            ],
        );
        assert_eq!(
            extract_model_from_multipart(&body, &ct),
            Some("/storage/models/zai-org/GLM-5-FP8".to_string())
        );
    }

    #[test]
    fn returns_none_when_no_model_field() {
        let (ct, body) = make_multipart("bnd", &[("prompt", "hello"), ("height", "480")]);
        assert_eq!(extract_model_from_multipart(&body, &ct), None);
    }

    #[test]
    fn returns_none_for_missing_boundary() {
        let body = Bytes::from("anything");
        assert_eq!(
            extract_model_from_multipart(&body, "multipart/form-data"),
            None
        );
    }

    #[test]
    fn quoted_boundary_in_content_type() {
        let ct = "multipart/form-data; boundary=\"WebKitBnd\"";
        let mut body_str = String::new();
        body_str.push_str("--WebKitBnd\r\n");
        body_str.push_str("Content-Disposition: form-data; name=\"model\"\r\n\r\n");
        body_str.push_str("my-model\r\n");
        body_str.push_str("--WebKitBnd--\r\n");
        let body = Bytes::from(body_str);
        assert_eq!(
            extract_model_from_multipart(&body, ct),
            Some("my-model".to_string())
        );
    }
}

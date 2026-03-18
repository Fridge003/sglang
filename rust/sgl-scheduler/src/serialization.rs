use crate::types::{FORMAT_PREFIX_MSGPACK_EMBEDDING, FORMAT_PREFIX_MSGPACK_TOKEN};

/// Serialize a BatchTokenIDOutput message to bytes with format prefix.
pub fn serialize_token_batch(data: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(1 + data.len());
    buf.push(FORMAT_PREFIX_MSGPACK_TOKEN);
    buf.extend_from_slice(data);
    buf
}

/// Serialize a BatchEmbeddingOutput message to bytes with format prefix.
pub fn serialize_embedding_batch(data: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(1 + data.len());
    buf.push(FORMAT_PREFIX_MSGPACK_EMBEDDING);
    buf.extend_from_slice(data);
    buf
}

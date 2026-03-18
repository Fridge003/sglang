/// Internal message type for the background sender thread.
pub enum OutputMessage {
    /// Pre-serialized msgpack bytes for a BatchTokenIDOutput
    TokenBatch(Vec<u8>),
    /// Pre-serialized msgpack bytes for a BatchEmbeddingOutput
    EmbeddingBatch(Vec<u8>),
    /// Signal to shut down the sender thread
    Shutdown,
}

/// Serialization format prefix bytes.
/// The first byte of each ZMQ message indicates the format:
/// - 0x00: pickle (legacy Python format)
/// - 0x01: msgpack for BatchTokenIDOutput
/// - 0x02: msgpack for BatchEmbeddingOutput
pub const FORMAT_PREFIX_PICKLE: u8 = 0x00;
pub const FORMAT_PREFIX_MSGPACK_TOKEN: u8 = 0x01;
pub const FORMAT_PREFIX_MSGPACK_EMBEDDING: u8 = 0x02;

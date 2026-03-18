use crate::serialization::{serialize_embedding_batch, serialize_token_batch};
use crate::types::OutputMessage;
use crossbeam_channel::Receiver;

/// Background thread that dequeues messages and sends them over ZMQ.
/// This runs entirely off the Python GIL.
pub fn zmq_sender_loop(rx: Receiver<OutputMessage>, endpoint: String, sndbuf_size: i32) {
    let ctx = zmq::Context::new();
    let socket = ctx.socket(zmq::PUSH).expect("Failed to create ZMQ PUSH socket");

    if sndbuf_size >= 0 {
        socket
            .set_sndbuf(sndbuf_size)
            .expect("Failed to set ZMQ SNDBUF");
    }

    // Set linger to 0 so socket doesn't block on shutdown
    socket.set_linger(0).expect("Failed to set ZMQ linger");

    socket
        .connect(&endpoint)
        .unwrap_or_else(|e| panic!("Failed to connect ZMQ socket to {}: {}", endpoint, e));

    loop {
        match rx.recv() {
            Ok(OutputMessage::TokenBatch(data)) => {
                let msg = serialize_token_batch(&data);
                if let Err(e) = socket.send(&msg, 0) {
                    eprintln!("ZMQ send error (token batch): {}", e);
                }
            }
            Ok(OutputMessage::EmbeddingBatch(data)) => {
                let msg = serialize_embedding_batch(&data);
                if let Err(e) = socket.send(&msg, 0) {
                    eprintln!("ZMQ send error (embedding batch): {}", e);
                }
            }
            Ok(OutputMessage::Shutdown) | Err(_) => {
                break;
            }
        }
    }
}

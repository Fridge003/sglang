use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use tokio::sync::Notify;
use tokio_stream::Stream;
use tonic::{Request, Response, Status};

use crate::bridge::{PyBridge, ResponseChunk};
use crate::proto;

pub struct SglangServiceImpl {
    pub bridge: Arc<PyBridge>,
    pub shutdown: Arc<Notify>,
}

fn sampling_params_to_json(params: &Option<proto::SamplingParams>) -> String {
    match params {
        Some(p) => {
            let mut map = serde_json::Map::new();
            if let Some(v) = p.temperature {
                map.insert("temperature".into(), serde_json::json!(v));
            }
            if let Some(v) = p.top_p {
                map.insert("top_p".into(), serde_json::json!(v));
            }
            if let Some(v) = p.top_k {
                map.insert("top_k".into(), serde_json::json!(v));
            }
            if let Some(v) = p.min_p {
                map.insert("min_p".into(), serde_json::json!(v));
            }
            if let Some(v) = p.frequency_penalty {
                map.insert("frequency_penalty".into(), serde_json::json!(v));
            }
            if let Some(v) = p.presence_penalty {
                map.insert("presence_penalty".into(), serde_json::json!(v));
            }
            if let Some(v) = p.repetition_penalty {
                map.insert("repetition_penalty".into(), serde_json::json!(v));
            }
            if let Some(v) = p.max_new_tokens {
                map.insert("max_new_tokens".into(), serde_json::json!(v));
            }
            if let Some(v) = p.min_new_tokens {
                map.insert("min_new_tokens".into(), serde_json::json!(v));
            }
            if !p.stop.is_empty() {
                map.insert("stop".into(), serde_json::json!(p.stop));
            }
            if !p.stop_token_ids.is_empty() {
                map.insert("stop_token_ids".into(), serde_json::json!(p.stop_token_ids));
            }
            if let Some(v) = p.ignore_eos {
                map.insert("ignore_eos".into(), serde_json::json!(v));
            }
            if let Some(v) = p.n {
                map.insert("n".into(), serde_json::json!(v));
            }
            if let Some(ref v) = p.json_schema {
                map.insert("json_schema".into(), serde_json::json!(v));
            }
            if let Some(ref v) = p.regex {
                map.insert("regex".into(), serde_json::json!(v));
            }
            serde_json::Value::Object(map).to_string()
        }
        None => "{}".to_string(),
    }
}

fn grpc_metadata_to_trace_headers(
    headers: &HashMap<String, String>,
) -> Option<HashMap<String, String>> {
    if headers.is_empty() {
        None
    } else {
        Some(headers.clone())
    }
}

type StreamResult<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

#[tonic::async_trait]
impl proto::sglang_service_server::SglangService for SglangServiceImpl {
    type TextGenerateStream = StreamResult<proto::TextGenerateResponse>;

    async fn text_generate(
        &self,
        request: Request<proto::TextGenerateRequest>,
    ) -> Result<Response<Self::TextGenerateStream>, Status> {
        let req = request.into_inner();
        let rid = req.rid.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let stream_flag = req.stream.unwrap_or(false);
        let return_logprob = req.return_logprob.unwrap_or(false);
        let top_logprobs_num = req.top_logprobs_num.unwrap_or(0);
        let logprob_start_len = req.logprob_start_len.unwrap_or(-1);
        let return_text_in_logprobs = req.return_text_in_logprobs.unwrap_or(false);
        let sampling_json = sampling_params_to_json(&req.sampling_params);
        let trace = grpc_metadata_to_trace_headers(&req.trace_headers);

        let receiver = self
            .bridge
            .submit_text_generate(
                &rid,
                &req.text,
                &sampling_json,
                stream_flag,
                return_logprob,
                top_logprobs_num,
                logprob_start_len,
                return_text_in_logprobs,
                req.lora_path.as_deref(),
                req.routing_key.as_deref(),
                req.routed_dp_rank,
                trace,
            )
            .map_err(|e| Status::internal(format!("Failed to submit request: {}", e)))?;

        let bridge = self.bridge.clone();
        let rid_clone = rid.clone();

        let stream = async_stream::stream! {
            loop {
                let chunk = tokio::task::spawn_blocking({
                    let receiver = receiver.clone();
                    move || receiver.recv()
                })
                .await
                .map_err(|e| Status::internal(format!("Task join error: {}", e)))?;

                match chunk {
                    Ok(ResponseChunk::Data(data)) => {
                        yield Ok(proto::TextGenerateResponse {
                            text: data.text.unwrap_or_default(),
                            meta_info: data.meta_info,
                            finished: false,
                        });
                    }
                    Ok(ResponseChunk::Finished(data)) => {
                        yield Ok(proto::TextGenerateResponse {
                            text: data.text.unwrap_or_default(),
                            meta_info: data.meta_info,
                            finished: true,
                        });
                        break;
                    }
                    Ok(ResponseChunk::Error(msg)) => {
                        yield Err(Status::internal(msg));
                        break;
                    }
                    Err(_) => {
                        // Channel closed — abort the request
                        let _ = bridge.abort(&rid_clone);
                        break;
                    }
                }
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }

    type GenerateStream = StreamResult<proto::GenerateResponse>;

    async fn generate(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let req = request.into_inner();
        let rid = req.rid.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let stream_flag = req.stream.unwrap_or(false);
        let return_logprob = req.return_logprob.unwrap_or(false);
        let top_logprobs_num = req.top_logprobs_num.unwrap_or(0);
        let logprob_start_len = req.logprob_start_len.unwrap_or(-1);
        let sampling_json = sampling_params_to_json(&req.sampling_params);
        let trace = grpc_metadata_to_trace_headers(&req.trace_headers);

        let receiver = self
            .bridge
            .submit_generate(
                &rid,
                req.input_ids,
                &sampling_json,
                stream_flag,
                return_logprob,
                top_logprobs_num,
                logprob_start_len,
                req.lora_path.as_deref(),
                req.routing_key.as_deref(),
                req.routed_dp_rank,
                trace,
            )
            .map_err(|e| Status::internal(format!("Failed to submit request: {}", e)))?;

        let bridge = self.bridge.clone();
        let rid_clone = rid.clone();

        let stream = async_stream::stream! {
            loop {
                let chunk = tokio::task::spawn_blocking({
                    let receiver = receiver.clone();
                    move || receiver.recv()
                })
                .await
                .map_err(|e| Status::internal(format!("Task join error: {}", e)))?;

                match chunk {
                    Ok(ResponseChunk::Data(data)) => {
                        yield Ok(proto::GenerateResponse {
                            output_ids: data.output_ids.unwrap_or_default(),
                            meta_info: data.meta_info,
                            finished: false,
                        });
                    }
                    Ok(ResponseChunk::Finished(data)) => {
                        yield Ok(proto::GenerateResponse {
                            output_ids: data.output_ids.unwrap_or_default(),
                            meta_info: data.meta_info,
                            finished: true,
                        });
                        break;
                    }
                    Ok(ResponseChunk::Error(msg)) => {
                        yield Err(Status::internal(msg));
                        break;
                    }
                    Err(_) => {
                        let _ = bridge.abort(&rid_clone);
                        break;
                    }
                }
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }

    async fn text_embed(
        &self,
        request: Request<proto::TextEmbedRequest>,
    ) -> Result<Response<proto::TextEmbedResponse>, Status> {
        let req = request.into_inner();
        let rid = req.rid.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let trace = grpc_metadata_to_trace_headers(&req.trace_headers);

        let receiver = self
            .bridge
            .submit_text_embed(&rid, &req.text, req.routing_key.as_deref(), trace)
            .map_err(|e| Status::internal(format!("Failed to submit request: {}", e)))?;

        let chunk = tokio::task::spawn_blocking(move || receiver.recv())
            .await
            .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
            .map_err(|_| Status::internal("Channel closed before response"))?;

        match chunk {
            ResponseChunk::Data(data) | ResponseChunk::Finished(data) => {
                Ok(Response::new(proto::TextEmbedResponse {
                    embedding: data.embedding.unwrap_or_default(),
                    meta_info: data.meta_info,
                }))
            }
            ResponseChunk::Error(msg) => Err(Status::internal(msg)),
        }
    }

    async fn embed(
        &self,
        request: Request<proto::EmbedRequest>,
    ) -> Result<Response<proto::EmbedResponse>, Status> {
        let req = request.into_inner();
        let rid = req.rid.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let trace = grpc_metadata_to_trace_headers(&req.trace_headers);

        let receiver = self
            .bridge
            .submit_embed(&rid, req.input_ids, req.routing_key.as_deref(), trace)
            .map_err(|e| Status::internal(format!("Failed to submit request: {}", e)))?;

        let chunk = tokio::task::spawn_blocking(move || receiver.recv())
            .await
            .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
            .map_err(|_| Status::internal("Channel closed before response"))?;

        match chunk {
            ResponseChunk::Data(data) | ResponseChunk::Finished(data) => {
                Ok(Response::new(proto::EmbedResponse {
                    embedding: data.embedding.unwrap_or_default(),
                    meta_info: data.meta_info,
                }))
            }
            ResponseChunk::Error(msg) => Err(Status::internal(msg)),
        }
    }

    async fn health_check(
        &self,
        _request: Request<proto::HealthCheckRequest>,
    ) -> Result<Response<proto::HealthCheckResponse>, Status> {
        let healthy = self
            .bridge
            .health_check()
            .map_err(|e| Status::internal(format!("Health check failed: {}", e)))?;

        Ok(Response::new(proto::HealthCheckResponse { healthy }))
    }

    async fn get_model_info(
        &self,
        _request: Request<proto::GetModelInfoRequest>,
    ) -> Result<Response<proto::GetModelInfoResponse>, Status> {
        let json_info = self
            .bridge
            .get_model_info()
            .map_err(|e| Status::internal(format!("Failed to get model info: {}", e)))?;

        Ok(Response::new(proto::GetModelInfoResponse {
            model_path: String::new(),
            json_info,
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<proto::GetServerInfoRequest>,
    ) -> Result<Response<proto::GetServerInfoResponse>, Status> {
        let json_info = self
            .bridge
            .get_server_info()
            .map_err(|e| Status::internal(format!("Failed to get server info: {}", e)))?;

        Ok(Response::new(proto::GetServerInfoResponse { json_info }))
    }

    async fn abort(
        &self,
        request: Request<proto::AbortRequest>,
    ) -> Result<Response<proto::AbortResponse>, Status> {
        let req = request.into_inner();
        self.bridge
            .abort(&req.rid)
            .map_err(|e| Status::internal(format!("Failed to abort: {}", e)))?;

        Ok(Response::new(proto::AbortResponse { success: true }))
    }
}

/// Start the Tonic gRPC server on the given address.
/// Returns a handle that can be used to shut down the server.
pub async fn run_grpc_server(
    addr: std::net::SocketAddr,
    bridge: Arc<PyBridge>,
    shutdown: Arc<Notify>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let service = SglangServiceImpl {
        bridge,
        shutdown: shutdown.clone(),
    };

    let svc = proto::sglang_service_server::SglangServiceServer::new(service);

    tracing::info!("gRPC server listening on {}", addr);

    tonic::transport::Server::builder()
        .add_service(svc)
        .serve_with_shutdown(addr, async move {
            shutdown.notified().await;
            tracing::info!("gRPC server shutting down");
        })
        .await?;

    Ok(())
}

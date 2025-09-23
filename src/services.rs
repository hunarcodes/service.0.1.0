use crate::inference::InferenceJob;
use axum::{Json, extract::State, http::StatusCode};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tonic::{Request, Response, Status};

// ---- App State ----
// This struct holds the shared resources our handlers need.
pub struct AppState {
    pub inference_sender: mpsc::Sender<InferenceJob>,
}

// ---- gRPC Service ----
pub mod grpc_service {
    use super::*;
    use crate::generated::inference::{
        EmbeddingRequest, EmbeddingResponse, inferencer_server::Inferencer,
    };

    pub struct MyInferenceService {
        pub app_state: Arc<AppState>,
    }

    #[tonic::async_trait]
    impl Inferencer for MyInferenceService {
        async fn get_embedding(
            &self,
            request: Request<EmbeddingRequest>,
        ) -> Result<Response<EmbeddingResponse>, Status> {
            let (response_sender, response_receiver) = oneshot::channel();
            let job = InferenceJob {
                text: request.into_inner().text,
                response_sender: Some(response_sender),
            };

            if self.app_state.inference_sender.send(job).await.is_err() {
                return Err(Status::internal("Inference channel closed"));
            }

            match response_receiver.await {
                Ok(embedding) => {
                    let reply = EmbeddingResponse { embedding };
                    Ok(Response::new(reply))
                }
                Err(_) => Err(Status::internal("Failed to receive embedding from model")),
            }
        }
    }
}

// ---- HTTP Service (axum) ----
#[derive(Deserialize)]
pub struct HttpEmbedRequest {
    pub text: String,
}

#[derive(Serialize)]
pub struct HttpEmbedResponse {
    pub embedding: Vec<f32>,
}

pub async fn http_get_embedding(
    State(app_state): State<Arc<AppState>>,
    Json(payload): Json<HttpEmbedRequest>,
) -> Result<Json<HttpEmbedResponse>, StatusCode> {
    let (response_sender, response_receiver) = oneshot::channel();
    let job = InferenceJob {
        text: payload.text,
        response_sender: Some(response_sender),
    };

    if app_state.inference_sender.send(job).await.is_err() {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    match response_receiver.await {
        Ok(embedding) => Ok(Json(HttpEmbedResponse { embedding })),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

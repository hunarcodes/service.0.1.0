use tonic::{Request, Response, Status};
use crate::generated::inference::{
    inferencer_server::{Inferencer, InferencerServer},
    EmbeddingRequest, EmbeddingResponse,
};

// Your service struct (call it something clear, not "inferencer_server")
pub struct MyInferenceService;

#[tonic::async_trait]
impl Inferencer for MyInferenceService {
    async fn get_embedding(
        &self,
        request: Request<EmbeddingRequest>,
    ) -> Result<Response<EmbeddingResponse>, Status> {
        let input = request.into_inner().text; // field name is "text" in proto
        let reply = EmbeddingResponse {
            embedding: vec![1.0, 2.0, 3.0], // dummy embedding output
        };
        Ok(Response::new(reply))
    }
}

// Helper to build server
pub fn new_server() -> InferencerServer<MyInferenceService> {
    InferencerServer::new(MyInferenceService)
}

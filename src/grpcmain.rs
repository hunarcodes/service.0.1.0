mod backend;
pub mod generated; // exposes src/generated/mod.rs -> inference.rs

use tonic::transport::Server;
use tonic_reflection::server::Builder;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;
    let svc = backend::new_server();

    // Enable server reflection using your generated protos
    let reflection_svc = Builder::configure()
        .register_encoded_file_descriptor_set(generated::FILE_DESCRIPTOR_SET)
        .build_v1()?;

    println!("Server listening on {}", addr);

    Server::builder()
        .add_service(svc)
        .add_service(reflection_svc)
        .serve(addr)
        .await?;

    Ok(())
}

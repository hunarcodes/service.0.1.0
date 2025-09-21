fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = "src/generated"; // keep generated code here

    tonic_prost_build::configure()
        .build_server(true) // generate gRPC server code
        .build_client(true) // generate gRPC client code
        .out_dir(out_dir)   // place files in src/generated
        .compile_protos(&["proto/inference.proto"], &["proto"])?;

    println!("cargo:rerun-if-changed=proto/inference.proto");
    Ok(())
}

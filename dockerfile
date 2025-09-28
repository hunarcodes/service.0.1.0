# ------------------------------
# Stage 1: Builder
# ------------------------------
FROM rust:1.89 AS builder

WORKDIR /app

# Install build dependencies if needed (prost/tonic with protoc)
RUN apt-get update && apt-get install -y protobuf-compiler pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

# Copy Cargo files first for dependency caching
COPY Cargo.toml Cargo.lock ./

# Create dummy src for cargo to resolve deps
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Pre-build dependencies to leverage caching
RUN cargo build --release || true

# Now copy actual source code (including build.rs and .proto files)
COPY . .

# Build release binary (this triggers build.rs for protos)
RUN cargo build --release

# ------------------------------
# Stage 2: Runtime
# ------------------------------
FROM debian:bookworm-slim

WORKDIR /app


# Install runtime dependencies
RUN apt-get update && apt-get install -y wget ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy only the compiled binary (no target folder bloat)
COPY --from=builder /app/target/release/gemma-embedder-rust /app/gemma-embedder-rust

# Copy model download script
COPY download_models.sh /usr/local/bin/download_models.sh
RUN chmod +x /usr/local/bin/download_models.sh

#Environment variables
ENV MODEL_VARIANT=q4
ENV MAX_TOKENS=2048
ENV MAX_BATCH_SIZE=32
ENV MAX_WAIT_MS=5

# On container start: download tokenizer + correct model, then run service
CMD ["sh", "-c", "download_models.sh && MODEL_PATH=$(ls model/*.onnx) exec ./gemma-embedder-rust"]


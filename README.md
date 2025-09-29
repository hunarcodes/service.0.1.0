
# EmbeddingGemma Microservice

A lightweight, Rust-based embedding service for [Google’s EmbeddingGemma-300M](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX), served via **HTTP** and **gRPC**, packaged as a small (~176 MB) Docker image.  
Built on [ONNX Runtime](https://onnxruntime.ai/) with batching and configurable runtime parameters.

---

## Features
- 🚀 Written in Rust — fast and minimalistic
- 📦 Very small docker images.  
- ⚙️ Configurable via environment variables (`MODEL_PATH`, `MAX_TOKENS`, etc.).  
- 🌐 Dual endpoints: HTTP (Axum) + gRPC (Tonic).  
- 🧵 Batched inference with configurable max batch size and wait time.  

---

## Run with Docker
**CPU:**
```bash
docker run -it --rm \
  -p 3000:3000 \
  -p 50051:50051 \
  -e MODEL_VARIANT=q4f16 \
  -e MODEL_PATH=model/model_q4f16.onnx \
  -e MAX_TOKENS=2048 \
  -e MAX_BATCH_SIZE=32 \
  -e MAX_WAIT_MS=5 \
  garvw/gemma-embedder-rust:latest
```
**GPU:**
```bash
docker run -it --rm \
  --gpus all \
  -p 3000:3000 \
  -p 50051:50051 \
  -e MODEL_VARIANT=base \
  -e MODEL_PATH=model/model_base.onnx \
  -e MAX_TOKENS=2048 \
  -e MAX_BATCH_SIZE=32 \
  -e MAX_WAIT_MS=5 \
  garvw/gemma-embedder-rust:gpu
```
The container will download tokenizer + model files at startup using `download_models.sh`.

---

## HTTP API

**Endpoint:** `POST /v1/embed`  

**Request:**
```json
{
  "text": "hello world"
}
```

**Response:**
```json
{
  "embedding": [0.1234, -0.5678, ...]
}
```

Test with curl:

```bash
curl -X POST http://localhost:3000/v1/embed \
  -H "Content-Type: application/json" \
  -d '{"text":"hello world"}'
```

---

## gRPC API

**Service:** `inference.Inferencer`  
**Method:** `GetEmbedding`

Example with grpcurl:

```bash
grpcurl -plaintext \
  -d '{"text":"hello world"}' \
  localhost:50051 inference.Inferencer/GetEmbedding
```

---

## Environment Variables

| Variable       | Default    | Description                                      |
|----------------|------------|--------------------------------------------------|
| MODEL_PATH     | required   | Path to .onnx file (downloaded to model/).       |
| MAX_TOKENS     | 2048       | Max sequence length for tokenizer.               |
| MAX_BATCH_SIZE | 32         | Max requests per inference batch.                |
| MAX_WAIT_MS    | 5          | Wait time (ms) to fill a batch before running.   |
| MODEL_VARIANT  | base       | Which model variant to download (base, fp16, q4, no_gather_q4, q4f16, quantized). |

---

## Build from Source

```bash
cargo build --release
```

## Run Directly

```bash
MODEL_PATH=model/model_q4f16.onnx \
MAX_TOKENS=2048 \
MAX_BATCH_SIZE=32 \
MAX_WAIT_MS=5 \
./target/release/gemma-embedder-rust
```

---

## License

EmbeddingGemma weights are licensed under Google’s Gemma Terms of Use.  
This project provides a service wrapper — you must comply with Google’s terms if you use the models.

Of course. Based on all the work we did, the old README is out of date. Here is a completely revised and improved version that reflects the final, working state of your project.

It includes a new "Quick Start" section, clarifies the CPU vs. GPU images, removes redundant environment variables from the examples, and provides correct instructions for building from source.

-----

# Embedding Gemma Microservice

A lightweight, high-performance Rust-based microservice for **[Google’s EmbeddingGemma-300M](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX)**. It serves embeddings via both **HTTP** and **gRPC** and is packaged in small, optimized Docker images for both CPU and GPU.

Built on **[ONNX Runtime](https://onnxruntime.ai/)**, it features dynamic request batching to maximize throughput.

-----

## Quick Start

#### 🚀 GPU Version (Requires NVIDIA Container Toolkit)

```bash
docker run --gpus all -it --rm \
  -p 3000:3000 \
  -p 50051:50051 \
  -e MODEL_VARIANT=q4 \
  garvw/gemma-embedder:gpu
```

#### 💻 CPU Version

```bash
docker run -it --rm \
  -p 3000:3000 \
  -p 50051:50051 \
  -e MODEL_VARIANT=q4 \
  garvw/gemma-embedder:cpu
```

*The container will download the specified model variant on first startup.*

-----

## Features

  - 🚀 **High Performance:** Written in Rust for minimal overhead and memory safety.
  - 📦 **Optimized Docker Images:** Small, secure images for both CPU and GPU.
  - **Strict Execution:** The GPU image **requires** a CUDA-enabled GPU and will exit if one is not found—no silent fallbacks.
  - 🌐 **Dual Endpoints:** A simple JSON REST API (via Axum) and a high-performance gRPC endpoint (via Tonic).
  - ⚙️ **Configurable:** Easily configure batch size, token length, and model variant via environment variables.
  - 🧵 **Dynamic Batching:** Automatically batches incoming requests to maximize inference throughput.

-----

## Run with Docker

This service is designed to be run as a Docker container. Two separate images are provided.

### Image Tags

  - **`garvw/gemma-embedder:gpu`**: For systems with an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.
  - **`garvw/gemma-embedder:cpu`**: For CPU-only environments.

The `MODEL_VARIANT` environment variable controls which model weights are downloaded. The `q4` (4-bit quantized) variant is recommended for a good balance of performance and quality.

-----

## API Endpoints

### HTTP API

**Endpoint:** `POST /v1/embed`

**Request Body:**

```json
{
  "text": "hello world"
}
```

**Example with `curl`:**

```bash
curl -X POST http://localhost:3000/v1/embed \
  -H "Content-Type: application/json" \
  -d '{"text":"hello world"}'
```

-----

### gRPC API

**Service:** `inference.Inferencer`
**Method:** `GetEmbedding`

**Example with `grpcurl`:**

```bash
grpcurl -plaintext \
  -d '{"text":"hello world"}' \
  localhost:50051 inference.Inferencer/GetEmbedding
```

-----

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `EXECUTION_PROVIDER` | `cpu` | Set to `gpu` in the GPU Dockerfile to enforce GPU execution. |
| `MODEL_VARIANT` | `q4` | Which model variant to download. `q4` or `fp32` (full-precision) are recommended. |
| `MODEL_PATH` | (auto) | Set automatically by the startup script based on `MODEL_VARIANT`. |
| `MAX_TOKENS` | `2048` | Max sequence length for the tokenizer. |
| `MAX_BATCH_SIZE` | `32` | Max number of requests to group into a single inference batch. |
| `MAX_WAIT_MS` | `5` | Time (in ms) to wait to fill a batch before running inference. |

-----

## Build from Source

You can build the Docker images locally instead of pulling from Docker Hub.

#### Building the GPU Image

```bash
docker build -f Dockerfile.gpu -t gemma-embedder:gpu-local .
```

#### Building the CPU Image

```bash
docker build -f Dockerfile.cpu -t gemma-embedder:cpu-local .
```

-----

## License

The EmbeddingGemma model weights are licensed under **Google’s Gemma Terms of Use**. This project provides a service wrapper for the model, and by using the Docker images, you are responsible for complying with Google’s terms.
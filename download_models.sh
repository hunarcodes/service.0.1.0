#!/usr/bin/env bash
set -e

MODEL_VARIANT=${MODEL_VARIANT:-base}
DEST_DIR="model"
BASE_URL="https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main"

mkdir -p "$DEST_DIR"

echo ">>> Downloading tokenizer files..."
TOKENIZER_FILES=(
    "tokenizer_config.json"
    "tokenizer.model"
    "tokenizer.json"
    "special_tokens_map.json"
    "generation_config.json"
    "config.json"
    "added_tokens.json"
)
for f in "${TOKENIZER_FILES[@]}"; do
    wget -c --show-progress "$BASE_URL/$f" -O "$DEST_DIR/$f"
done

echo ">>> Selected model variant: $MODEL_VARIANT"

case "$MODEL_VARIANT" in
  base)
    MODEL_FILE="model.onnx"
    wget -c --show-progress "$BASE_URL/onnx/model.onnx" -O "$DEST_DIR/$MODEL_FILE"
    wget -c --show-progress "$BASE_URL/onnx/model.onnx_data" -O "$DEST_DIR/model.onnx_data"
    ;;
  fp16)
    MODEL_FILE="model_fp16.onnx"
    wget -c --show-progress "$BASE_URL/onnx/model_fp16.onnx" -O "$DEST_DIR/$MODEL_FILE"
    wget -c --show-progress "$BASE_URL/onnx/model_fp16.onnx_data" -O "$DEST_DIR/model_fp16.onnx_data"
    ;;
  q4)
    MODEL_FILE="model_q4.onnx"
    wget -c --show-progress "$BASE_URL/onnx/model_q4.onnx" -O "$DEST_DIR/$MODEL_FILE"
    wget -c --show-progress "$BASE_URL/onnx/model_q4.onnx_data" -O "$DEST_DIR/model_q4.onnx_data"
    ;;
  no_gather_q4)
    MODEL_FILE="model_no_gather_q4.onnx"
    wget -c --show-progress "$BASE_URL/onnx/model_no_gather_q4.onnx" -O "$DEST_DIR/$MODEL_FILE"
    wget -c --show-progress "$BASE_URL/onnx/model_no_gather_q4.onnx_data" -O "$DEST_DIR/model_no_gather_q4.onnx_data"
    ;;
  q4f16)
    MODEL_FILE="model_q4f16.onnx"
    wget -c --show-progress "$BASE_URL/onnx/model_q4f16.onnx" -O "$DEST_DIR/$MODEL_FILE"
    wget -c --show-progress "$BASE_URL/onnx/model_q4f16.onnx_data" -O "$DEST_DIR/model_q4f16.onnx_data"
    ;;
  quantized)
    MODEL_FILE="model_quantized.onnx"
    wget -c --show-progress "$BASE_URL/onnx/model_quantized.onnx" -O "$DEST_DIR/$MODEL_FILE"
    wget -c --show-progress "$BASE_URL/onnx/model_quantized.onnx_data" -O "$DEST_DIR/model_quantized.onnx_data"
    ;;
  *)
    echo "ERROR: Unknown MODEL_VARIANT '$MODEL_VARIANT'. Use: base, fp16, q4, no_gather_q4, q4f16, quantized."
    exit 1
    ;;
esac

# Export for downstream Rust binary
export MODEL_PATH="$DEST_DIR/$MODEL_FILE"
echo ">>> MODEL_PATH set to $MODEL_PATH"

echo ">>> All files downloaded into $DEST_DIR/"

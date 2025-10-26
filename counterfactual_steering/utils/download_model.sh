#!/bin/bash

# Download gpt-oss-20b model from HuggingFace
# This script downloads the model to a local directory for use with vLLM

# Activate virtual environment
source /root/counterfactual_steering/.venv/bin/activate

# Default configuration
MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-20b}"
LOCAL_DIR="${LOCAL_DIR:-/root/counterfactual_steering/models/gpt-oss-20b}"

echo "Downloading model: $MODEL_NAME"
echo "Local directory: $LOCAL_DIR"
echo ""

# Check if huggingface_hub is installed
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface_hub..."
    pip install -U "huggingface_hub[cli]"
fi

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Download the model
echo "Starting download (this may take a while)..."
huggingface-cli download "$MODEL_NAME" \
    --local-dir "$LOCAL_DIR" \
    --local-dir-use-symlinks False

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Model downloaded successfully!"
    echo "Local path: $LOCAL_DIR"
    echo ""
    echo "To use this model, set MODEL_PATH in start_vllm_server.sh:"
    echo "  export MODEL_PATH=\"$LOCAL_DIR\""
    echo "  ./start_vllm_server.sh"
else
    echo ""
    echo "✗ Download failed. Please check:"
    echo "  1. You have internet connection"
    echo "  2. You have enough disk space (model is ~40GB)"
    echo "  3. You have access to the model on HuggingFace"
    exit 1
fi


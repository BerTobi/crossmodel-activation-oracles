export HF_HOME=/workspace/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1
echo "KERNELS START $(date -u +%H:%M:%SZ)"
pip install -q --break-system-packages flash-linear-attention 2>&1 | tail -1; echo "fla exit $?"
pip install -q --break-system-packages causal-conv1d 2>&1 | tail -2; echo "causal-conv1d exit $?"
python3 -c "import fla; print(\"fla OK\", fla.__version__)" 2>&1 | tail -1
python3 -c "import causal_conv1d; print(\"causal_conv1d OK\")" 2>&1 | tail -1
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(\"Qwen/Qwen3.5-9B\", allow_patterns=[\"*.json\",\"*.safetensors\",\"*.txt\",\"*.model\",\"*.py\",\"tokenizer*\",\"*.tiktoken\",\"*.jinja\"]); print(\"9B downloaded\")"
echo "KERNELS_AND_9B_DONE $(date -u +%H:%M:%SZ)"

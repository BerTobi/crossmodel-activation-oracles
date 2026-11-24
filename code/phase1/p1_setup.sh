#!/usr/bin/env bash
# Phase 1 environment: the phase 0 pins that the organism work needs (torch 2.7.1, transformers 4.55.2, peft 0.17.1), the base model.
# Status lines -> /workspace/p1/status.
set -u
W=/workspace; K=$W/p1; S=$K/status
mkdir -p $K $W/hf-cache $W/uv-cache
st() { echo "$* $(date -u +%H:%M:%SZ)" >> $S; }
export HF_HOME=$W/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 UV_CACHE_DIR=$W/uv-cache UV_LINK_MODE=copy
st "SETUP START $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > $K/uv_install.log 2>&1
export PATH=$HOME/.local/bin:$PATH
PY=$W/venv/bin/python
if [ ! -x $PY ]; then
  uv venv --python 3.11 $W/venv > $K/uv_venv.log 2>&1 || { st "SETUP ABORT: uv venv"; exit 1; }
  uv pip install --python $PY "torch==2.7.1" > $K/uv_pip.log 2>&1 || st "note: torch install errors"
  uv pip install --python $PY "transformers==4.55.2" "peft==0.17.1" "datasets==4.8.4" "accelerate==1.10.1" "numpy<2" "pyarrow==21.0.0" \
    scikit-learn scipy "huggingface-hub>=0.30" hf_transfer tqdm >> $K/uv_pip.log 2>&1 || st "note: pinned install reported errors (uv_pip.log)"
fi
$PY - <<'PYEOF' > $K/download.log 2>&1
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3-8B")); print("downloads done")
PYEOF
grep -q "downloads done" $K/download.log || { st "SETUP ABORT: download failed"; exit 1; }
$PY - <<'PYEOF' > $K/import_check.log 2>&1
import torch, transformers, peft, datasets, sklearn
print("torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__, "datasets", datasets.__version__)
print("cuda ok", torch.cuda.is_available(), torch.cuda.get_device_name(0))
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
m = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}, {"role": "user", "content": "q2"}, {"role": "assistant", "content": "<think>\nNOTE\n</think>\n\nREPLY"}]
print("TEMPLATE_TRAIN:", repr(t.apply_chat_template(m, tokenize=False, enable_thinking=True)))
print("TEMPLATE_GEN_THINK:", repr(t.apply_chat_template(m[:1], tokenize=False, add_generation_prompt=True, enable_thinking=True)))
print("TEMPLATE_GEN_NOTHINK:", repr(t.apply_chat_template(m[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)))
print("imports OK")
PYEOF
st "env: $(grep -a -E '^torch|^cuda ok' $K/import_check.log | tr '\n' ' ' | cut -c1-200)"
grep -q "imports OK" $K/import_check.log && st "SETUP_OK" || st "SETUP FAILED (import_check.log)"

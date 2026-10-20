#!/usr/bin/env bash
# Environment for post-training the Qwen3.6-27B oracle on the organism backbone: torch 2.12.1 (Triton 3.7.1, needed for correct
# gated-DeltaNet backward on Hopper), transformers 5 / peft 0.21, the fast linear-attention kernels (REQUIRED for training by upstream
# sft.py), upstream main, the merged organism and the released oracle. Status lines -> /workspace/posttrain/status.
set -u
W=/workspace; K=$W/posttrain; S=$K/status
mkdir -p $K $W/models $W/hf-cache
st() { echo "$* $(date -u +%H:%M:%SZ)" >> $S; }
export HF_HOME=$W/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1
st "SETUP START $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1) nvcc=$(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' || echo none)"

AO_COMMIT=5712762
if [ ! -d $W/activation_oracles ]; then git clone -q https://github.com/adamkarvonen/activation_oracles.git $W/activation_oracles; fi
(cd $W/activation_oracles && git checkout -q $AO_COMMIT) || { st "SETUP ABORT: cannot checkout $AO_COMMIT"; exit 1; }

# downloads first (they overlap with the pip work below only if run in the background; keep it simple and sequential)
python3 - <<'EOF' > $K/download.log 2>&1
from huggingface_hub import snapshot_download
snapshot_download("ai-safety-institute/Qwen3.6-27B-eval_sandbagger-merged", local_dir="/workspace/models/organism_merged")
snapshot_download("adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3.6-27B", local_dir="/workspace/models/oracle")
print("downloads done")
EOF
grep -q "downloads done" $K/download.log || { st "SETUP ABORT: download failed"; exit 1; }
st "downloads: $(du -sh $W/models/organism_merged $W/models/oracle | awk '{print $1}' | tr '\n' ' ')"

pip install -q "torch==2.12.1" --index-url https://download.pytorch.org/whl/cu128 > $K/pip.log 2>&1 || st "note: torch 2.12.1 cu128 install failed"
pip install -q "transformers>=5.17" "peft==0.21.0" "datasets==3.6.0" accelerate hf_transfer safetensors pandas pyarrow fastparquet "pydantic==2.11.7" tqdm einops wandb codetiming jaxtyping fire slist regex rapidfuzz python-dotenv dill hydra-core pylatexenc pybind11 packaging >> $K/pip.log 2>&1
pip install -q "flash-linear-attention==0.5.2" >> $K/pip.log 2>&1 || st "note: fla install failed"
MAX_JOBS=16 pip install -q --no-build-isolation "causal-conv1d==1.7.0" >> $K/pip.log 2>&1 || st "note: causal-conv1d build failed (training needs it)"
pip install -q --no-deps -e $W/activation_oracles >> $K/pip.log 2>&1 || true
python3 - <<'EOF' >> $K/pip.log 2>&1 || st "SETUP ABORT: import check failed (see pip.log)"
import torch, transformers, peft, datasets, triton
print("torch", torch.__version__, "triton", triton.__version__, "transformers", transformers.__version__, "peft", peft.__version__, "datasets", datasets.__version__)
print("cuda ok", torch.cuda.is_available(), torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
import fla; print("fla", fla.__version__)
try:
    import causal_conv1d; print("causal_conv1d", causal_conv1d.__version__)
except Exception as e:
    print("causal_conv1d MISSING:", e)
import sys; sys.path.insert(0, "/workspace/activation_oracles")
from nl_probes import sft
print("nl_probes.sft imports OK")
EOF
st "env: $(grep -a -E '^torch|^cuda ok|^fla|^causal|imports OK' $K/pip.log | tr '\n' ' ' | cut -c1-300)"
grep -q "imports OK" $K/pip.log && st "SETUP_OK" || st "SETUP FAILED"

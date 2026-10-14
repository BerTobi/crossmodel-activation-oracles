#!/usr/bin/env bash
# Pod bootstrap for the released-pair experiment (Qwen3.6-27B: AISI eval_sandbagger organism + Karvonen oracle). One H100 80GB.
# Usage on the pod:  bash /workspace/pair27b/pod_setup_pair.sh        (idempotent; status lines -> /workspace/pair27b/status)
set -u
W=/workspace; P=$W/pair27b; S=$P/status
mkdir -p $P $W/models $W/hf-cache
st() { echo "$* $(date -u +%H:%M:%SZ)" >> $S; }
export HF_HOME=$W/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false PIP_DISABLE_PIP_VERSION_CHECK=1
st "SETUP START $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# --- deps: upstream main (transformers 5 era) without its pinned torch; fast kernels for the Gated DeltaNet layers if they build
AO_COMMIT=5712762
if [ ! -d $W/activation_oracles ]; then git clone -q https://github.com/adamkarvonen/activation_oracles.git $W/activation_oracles; fi
(cd $W/activation_oracles && git checkout -q $AO_COMMIT) || { st "SETUP ABORT: cannot checkout $AO_COMMIT"; exit 1; }
pip install -q "transformers>=5.17" "peft>=0.21" accelerate datasets hf_transfer safetensors pandas pyarrow pydantic tqdm einops > $P/pip.log 2>&1
pip install -q flash-linear-attention >> $P/pip.log 2>&1 || st "note: flash-linear-attention install failed (slow fallback)"
pip install -q causal-conv1d >> $P/pip.log 2>&1 || st "note: causal-conv1d install failed (slow fallback)"
pip install -q --no-deps -e $W/activation_oracles >> $P/pip.log 2>&1 || st "note: editable install of activation_oracles failed; using PYTHONPATH"
python3 - <<'EOF' >> $P/pip.log 2>&1 || st "SETUP ABORT: import check failed (see pip.log)"
import transformers, peft, torch
print("transformers", transformers.__version__, "peft", peft.__version__, "torch", torch.__version__)
import sys; sys.path.insert(0, "/workspace/activation_oracles")
from nl_probes.base_experiment import VerbalizerEvalConfig
from nl_probes.utils.activation_utils import get_hf_submodule
print("activation_oracles imports OK")
EOF
st "deps: $(tail -2 $P/pip.log | tr '\n' ' ' | cut -c1-120)"

# --- weights (all public, no token): base 54GB, organism adapter 2.5GB, oracle adapter 1.9GB
python3 - <<'EOF' >> $P/download.log 2>&1
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.6-27B", allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.py"])
snapshot_download("ai-safety-institute/Qwen3.6-27B-eval_sandbagger", local_dir="/workspace/models/organism")
snapshot_download("adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3.6-27B", local_dir="/workspace/models/oracle")
print("downloads done")
EOF
grep -q "downloads done" $P/download.log || { st "SETUP ABORT: download failed"; exit 1; }
st "SETUP_OK models: $(du -sh $W/hf-cache/hub/models--Qwen--Qwen3.6-27B $W/models/organism $W/models/oracle 2>/dev/null | awk '{print $1}' | tr '\n' ' ')"

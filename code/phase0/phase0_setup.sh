#!/usr/bin/env bash
# Phase 0 environment: Bauer et al.'s trainer (japhba/activation_oracles @ bf74e64) and AObench (ceselder/cot-oracle @ fb28b55,
# the commit whose saved eval_results are our reference), one venv built from the trainer's own uv.lock (torch 2.7.1,
# transformers 4.55.2, peft 0.17.1, vllm 0.10.0) with a pinned fallback, the base model and the three released adapters.
# Status lines -> /workspace/phase0/status.
set -u
W=/workspace; K=$W/phase0; S=$K/status
mkdir -p $K $W/hf-cache $W/uv-cache
st() { echo "$* $(date -u +%H:%M:%SZ)" >> $S; }
export HF_HOME=$W/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 PIP_BREAK_SYSTEM_PACKAGES=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export UV_PROJECT_ENVIRONMENT=$W/venv UV_CACHE_DIR=$W/uv-cache UV_LINK_MODE=copy
st "SETUP START $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"

AO_COMMIT=bf74e64; COT_COMMIT=fb28b55
[ -d $W/ao ] || git clone -q https://github.com/japhba/activation_oracles.git $W/ao
(cd $W/ao && git checkout -q $AO_COMMIT) || { st "SETUP ABORT: trainer checkout $AO_COMMIT"; exit 1; }
[ -d $W/cot-oracle ] || git clone -q https://github.com/ceselder/cot-oracle.git $W/cot-oracle
(cd $W/cot-oracle && git checkout -q $COT_COMMIT) || { st "SETUP ABORT: AObench checkout $COT_COMMIT"; exit 1; }
st "repos: ao $(cd $W/ao && git rev-parse --short HEAD) cot-oracle $(cd $W/cot-oracle && git rev-parse --short HEAD)"

command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > $K/uv_install.log 2>&1
export PATH=$HOME/.local/bin:$PATH
st "uv $(uv --version 2>/dev/null || echo missing)"
PY=$W/venv/bin/python

# their uv.lock predates the eval/data extras in pyproject (and has no unsloth), so the key versions it pins are installed by hand:
# torch 2.7.1, transformers 4.55.2, peft 0.17.1, trl 0.18.2, datasets 4.8.4, accelerate 1.10.1, bitsandbytes 0.48.1, vllm 0.10.0, xformers 0.0.31
cd $W/ao
if [ ! -x $PY ]; then
  uv venv --python 3.11 $W/venv > $K/uv_venv.log 2>&1 || { st "SETUP ABORT: uv venv (uv_venv.log)"; exit 1; }
  uv pip install --python $PY "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" > $K/uv_pip.log 2>&1 || st "note: torch install errors (uv_pip.log)"
  uv pip install --python $PY "transformers==4.55.2" "peft==0.17.1" "trl==0.18.2" "datasets==4.8.4" "accelerate==1.10.1" "bitsandbytes==0.48.1" \
    "vllm==0.10.0" "xformers==0.0.31" "numpy<2" "pydantic>=2.10" python-dotenv wandb tqdm "huggingface-hub>=0.30" hf_transfer "pyarrow==21.0.0" \
    scikit-learn scipy anthropic openai tenacity slist rapidfuzz math_verify latex2sympy2_extended matplotlib httpx einops "ray[default]" hydra-core \
    "tensordict>=0.8" fastparquet pandas packaging >> $K/uv_pip.log 2>&1 || st "note: pinned install reported errors (uv_pip.log)"
fi
uv pip install --python $PY --no-deps -e $W/ao > $K/uv_pip_ao.log 2>&1 || st "note: editable install of the trainer failed (uv_pip_ao.log)"

# flash-attn: prebuilt wheel only (a source build takes an hour); AObench falls back to sdpa, the trainer via AO_ATTN
if ! $PY -c "import flash_attn" > /dev/null 2>&1; then
  TV=$($PY -c "import torch;print(torch.__version__.split('+')[0][:3])")
  ABI=$($PY -c "import torch;print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
  PYTAG=$($PY -c "import sys;print(f'cp{sys.version_info[0]}{sys.version_info[1]}')")
  for abi in $ABI $([ "$ABI" = TRUE ] && echo FALSE || echo TRUE); do
    url="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch${TV}cxx11abi${abi}-${PYTAG}-${PYTAG}-linux_x86_64.whl"
    uv pip install --python $PY "$url" > $K/fa.log 2>&1 && $PY -c "import flash_attn" > /dev/null 2>&1 && { st "flash-attn wheel ok (abi $abi)"; break; }
  done
fi
$PY -c "import flash_attn" > /dev/null 2>&1 || st "note: flash-attn unavailable -> sdpa"

# unsloth is in their pyproject but not in their lock: try it with the lock's core versions held fixed, otherwise plain PEFT
printf 'torch==2.7.1\ntransformers==4.55.2\npeft==0.17.1\ntrl==0.18.2\n' > $K/constraints.txt
if ! $PY -c "import unsloth" > /dev/null 2>&1; then
  timeout 900 uv pip install --python $PY -c $K/constraints.txt "unsloth>=2026.5" > $K/unsloth.log 2>&1 \
    && $PY -c "import unsloth" > $K/unsloth_import.log 2>&1 && st "unsloth ok" || st "note: unsloth not usable (unsloth.log / unsloth_import.log); smoke uses plain PEFT"
fi

# the trainer's load_model hard-codes flash_attention_2; let AO_ATTN override it
$PY - <<'PYEOF' >> $S
from pathlib import Path
p = Path("/workspace/ao/nl_probes/utils/common.py"); s = p.read_text()
old = 'attn = "eager" if "gemma" in model_name.lower() else "flash_attention_2"'
if old in s:
    s = s.replace(old, 'attn = "eager" if "gemma" in model_name.lower() else __import__("os").environ.get("AO_ATTN", "flash_attention_2")')
    p.write_text(s); print("common.py: AO_ATTN override added")
else:
    print("common.py: anchor missing or already patched")
PYEOF

$PY - <<'PYEOF' > $K/download.log 2>&1
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-8B")
for r in ["ceselder/qwen3-8b-ao-v3-best", "ceselder/qwen3-8b-ao-v3-adam-baseline", "ceselder/adam-reupload-qwen3-8b-latentqa-cls-past-lens"]:
    print(r, snapshot_download(r))
print("downloads done")
PYEOF
grep -q "downloads done" $K/download.log || { st "SETUP ABORT: download failed"; exit 1; }
st "downloads: $(du -sh $W/hf-cache | cut -f1)"

$PY - <<'PYEOF' > $K/import_check.log 2>&1
import torch, transformers, peft, datasets
print("torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__, "datasets", datasets.__version__)
print("cuda ok", torch.cuda.is_available(), torch.cuda.get_device_name(0))
import sys; sys.path.insert(0, "/workspace/ao")
from nl_probes import sft
print("trainer imports OK")
sys.path.insert(0, "/workspace/cot-oracle")
from AObench.open_ended_eval import run_all
from AObench.utils.report import generate_report
print("AObench imports OK")
PYEOF
st "env: $(grep -a -E '^torch|^cuda ok|imports OK' $K/import_check.log | tr '\n' ' ' | cut -c1-300)"
grep -q "AObench imports OK" $K/import_check.log && st "SETUP_OK" || st "SETUP FAILED (import_check.log)"

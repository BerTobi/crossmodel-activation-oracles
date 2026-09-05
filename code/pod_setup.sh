#!/usr/bin/env bash
# One-shot setup for a fresh RunPod pod (runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404).
# Everything lives under /workspace (the pod's persistent volume). Container disk is wiped on stop.
#
#   bash pod_setup.sh            # prep + smoke test
#   bash pod_setup.sh --no-smoke # prep only
#
# HF access: only lmsys/lmsys-chat-1m (past-lens data) is gated. Both models are ungated.
#   - If logged in (`huggingface-cli login` once on the pod, or HF_TOKEN exported) AND lmsys access is
#     granted, the smoke test includes past-lens. Otherwise it runs with AO_SMOKE_NO_PASTLENS=1 and
#     still exercises the entire cross-model path (both models, capture, LoRA train, eval).
#   - WANDB_API_KEY optional; otherwise wandb runs offline.
set -euo pipefail

export HF_HOME=/workspace/hf-cache          # model + dataset cache on the VOLUME (survives stop/start)
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
PIP="pip install -q --break-system-packages"   # PEP 668: install into the template's system torch
cd /workspace
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "== python deps (training path only; no vllm/ray/flash-attn) =="
$PIP -U huggingface_hub hf_transfer
$PIP "transformers==4.55.2" "peft==0.17.1" "accelerate==1.10.1" "datasets==3.6.0" sentencepiece protobuf \
     "pydantic==2.11.7" "einops==0.8.1" "wandb==0.21.1" "bitsandbytes==0.48.1" \
     pandas pyarrow fastparquet tqdm python-dotenv "numpy<2.0.0" \
     anthropic openai tenacity slist rapidfuzz fire typer jaxtyping regex   # imported transitively via sae_training_data -> autointerp caller
if python3 -c "import flash_attn" 2>/dev/null; then echo "flash-attn: present"; AO_ATTN=""; else echo "flash-attn: absent -> AO_ATTN=sdpa"; AO_ATTN=sdpa; fi

echo "== code =="
if [[ ! -d activation_oracles ]]; then git clone --depth 1 https://github.com/adamkarvonen/activation_oracles.git; fi
if ! grep -q "subject_model_name" activation_oracles/nl_probes/sft.py; then
  python3 "$SCRIPT_DIR/crossmodel_patch.py" activation_oracles      # fresh clone: full patch (incl. injection recipe)
elif ! grep -q "injection_mode" activation_oracles/nl_probes/sft.py; then
  python3 "$SCRIPT_DIR/apply_injection_patch.py" activation_oracles /dev/null   # existing clone: add injection recipe
fi
if ! grep -q "oracle_hidden_size" activation_oracles/nl_probes/utils/dataset_utils.py; then
  python3 "$SCRIPT_DIR/apply_padding_patch.py" activation_oracles /dev/null     # existing clone: add zero-padding
fi
if ! grep -q "is_std" activation_oracles/nl_probes/utils/activation_utils.py; then
  python3 "$SCRIPT_DIR/apply_repro_patch.py" activation_oracles /dev/null       # existing clone: local-path fix + AO_REVISION
fi
if ! grep -q "subject_lora_path" activation_oracles/nl_probes/dataset_classes/classification.py; then
  python3 "$SCRIPT_DIR/apply_subjectlora_patch.py" activation_oracles /dev/null # existing clone: adapter-aware dataset build
fi
if ! grep -q "AO_ORACLE_REVISION" activation_oracles/nl_probes/sft.py; then
  python3 "$SCRIPT_DIR/apply_revision_fix.py" activation_oracles /dev/null    # existing clone: per-model HF revision
fi
if ! grep -q "pad_precomputed" activation_oracles/nl_probes/utils/dataset_utils.py; then
  python3 "$SCRIPT_DIR/apply_padding_fix.py" activation_oracles /dev/null      # existing clone: pad precomputed vectors
fi
if grep -q "hook_layer = hook_layer_abs" activation_oracles/nl_probes/sft.py; then
  python3 "$SCRIPT_DIR/apply_hooklayer_fix.py" activation_oracles /dev/null    # existing clone: hook_layer definition-order fix
fi
if grep -q "truncation is lossy and not implemented" activation_oracles/nl_probes/sft.py; then
  python3 "$SCRIPT_DIR/apply_truncation_patch.py" activation_oracles /dev/null  # existing clone: truncation (subject wider than oracle)
fi
echo "patch state: cross-model=$(grep -c subject_model_name activation_oracles/nl_probes/sft.py) injection=$(grep -c injection_mode activation_oracles/nl_probes/sft.py) padding=$(grep -c oracle_hidden_size activation_oracles/nl_probes/utils/dataset_utils.py) repro=$(grep -c is_std activation_oracles/nl_probes/utils/activation_utils.py) subject_lora=$(grep -c subject_lora_path activation_oracles/nl_probes/dataset_classes/classification.py)"
cd activation_oracles
$PIP -e . --no-deps
git diff --stat | tail -1

ORACLE="${AO_ORACLE_MODEL:-Qwen/Qwen3-14B}"          # sprint choice: Qwen3-14B oracle (d=5120) reading Qwen3-8B (d=4096, zero-padded)
SUBJECT="${AO_SUBJECT_MODEL:-Qwen/Qwen3-8B}"
echo "== pre-download oracle=$ORACLE subject=$SUBJECT to the volume =="
huggingface-cli download "$ORACLE" --exclude "original/*" >/dev/null
huggingface-cli download "$SUBJECT" --revision "${AO_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}" >/dev/null
python3 - "$ORACLE" "$SUBJECT" <<'PYC'
import sys
from transformers import AutoConfig
o = AutoConfig.from_pretrained(sys.argv[1]); s = AutoConfig.from_pretrained(sys.argv[2])
mode = "zero-pad" if s.hidden_size < o.hidden_size else ("TRUNCATE (lossy)" if s.hidden_size > o.hidden_size else "equal width")
pad = abs(o.hidden_size - s.hidden_size)
print(f"oracle {sys.argv[1]}: d={o.hidden_size} layers={o.num_hidden_layers} | subject {sys.argv[2]}: d={s.hidden_size} layers={s.num_hidden_layers} | {mode} {pad} dims")
PYC
du -sh /workspace/hf-cache 2>/dev/null | tail -1

# ---------------- HF access detection ----------------
if [[ -n "${HF_TOKEN:-}" ]]; then huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential >/dev/null; fi
HAS_LMSYS=0
if huggingface-cli whoami >/dev/null 2>&1; then
  if python3 -c 'from datasets import load_dataset; next(iter(load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)))' >/dev/null 2>&1; then
    HAS_LMSYS=1
  fi
fi
if [[ $HAS_LMSYS == 1 ]]; then echo "HF: logged in with lmsys access -> past-lens ENABLED"; else echo "HF: no login or no lmsys access -> past-lens OFF for smoke (AO_SMOKE_NO_PASTLENS=1)"; fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then WANDB_MODE=offline; echo "wandb: offline"; else WANDB_MODE=""; wandb login "$WANDB_API_KEY" >/dev/null; fi

# Environment for all later shells
cat > /workspace/env.sh <<EOF
export HF_HOME=/workspace/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
export AO_ORACLE_MODEL="${ORACLE}"
export AO_SUBJECT_MODEL="${SUBJECT}"
# The organism (Taboo subject) adapter, loaded onto the subject and kept ACTIVE during capture. Copy it to the pod first:
#   scp -r "prev paper data/extracted/Desktop/Claude/Misaligned-Oracles/clock_bundle/Mpp_clock_adapter" root@<pod>:/workspace/
export AO_SUBJECT_LORA="${AO_SUBJECT_LORA:-/workspace/Mpp_clock_adapter}"
export AO_ATTN="${AO_ATTN}"
export WANDB_MODE="${WANDB_MODE}"
# Injection recipe. Sprint default = Karvonen (norm_matched @ oracle layer 1) so the released Qwen3-8B self-oracle is a
# recipe-matched anchor and zero-padding is scale-safe.
# Bersia & Gaintseva App. B.3 states: raw lambda=1.0 at ABSOLUTE AO layer 18 (for a Qwen3-8B oracle). To replicate that:
#   AO_INJECTION=raw AO_HOOK_LAYER=18.  Mapping "18" to a different-depth oracle is a DESIGN CHOICE the paper does not make;
#   AO_HOOK_LAYER_PERCENT=50 is one option (Qwen3-8B L18 -> Qwen3-14B L20 -> Llama L16), keeping the absolute index is another.
export AO_INJECTION="${AO_INJECTION:-norm_matched}"
export AO_HOOK_LAYER="${AO_HOOK_LAYER:-1}"
export AO_HOOK_LAYER_PERCENT="${AO_HOOK_LAYER_PERCENT:-}"
export AO_LAMBDA="${AO_LAMBDA:-1.0}"
# AO_REVISION pins the SUBJECT only (the Qwen3-8B commit the released oracle was trained on; handoff section 6).
# The oracle is unpinned unless AO_ORACLE_REVISION is set (a hash from one repo does not exist in another).
export AO_REVISION="${AO_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
export TORCHDYNAMO_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF
source /workspace/env.sh
if [[ -n "${AO_SUBJECT_LORA:-}" ]]; then
  if [[ -f "$AO_SUBJECT_LORA/adapter_config.json" ]]; then echo "subject adapter: $AO_SUBJECT_LORA ($(stat -c %s "$AO_SUBJECT_LORA/adapter_model.safetensors") bytes)"; else echo "WARNING: AO_SUBJECT_LORA=$AO_SUBJECT_LORA not found on the volume"; fi
fi
echo "== prep complete =="

if [[ "${1:-}" != "--no-smoke" ]]; then
  SMOKE_NO_PASTLENS=$(( 1 - HAS_LMSYS ))
  echo "== SMOKE TEST (AO_DEBUG=1: ~100 examples/dataset, tiny eval splits; past-lens $([[ $HAS_LMSYS == 1 ]] && echo ON || echo OFF)) =="
  echo "   expect: dataset build (loads Qwen3-8B once for the classification test split), then a few train steps + eval"
  set +e
  AO_DEBUG=1 AO_SMOKE_NO_PASTLENS=$SMOKE_NO_PASTLENS torchrun --nproc_per_node=1 nl_probes/sft.py > /workspace/smoke.log 2>&1
  RC=$?
  set -e
  tail -40 /workspace/smoke.log
  echo "== SMOKE TEST FINISHED (exit $RC); full log at /workspace/smoke.log =="
  echo "SMOKE_EXIT=$RC" > /workspace/smoke.status
fi

cat <<'EOF'

============================================================
 FULL RUN (~1M examples, 1 epoch; ~10-12 h on H100 SXM) — launch DETACHED so it survives SSH drop
============================================================
  source /workspace/env.sh && cd /workspace/activation_oracles
  setsid bash -c "torchrun --nproc_per_node=1 nl_probes/sft.py" > /workspace/train.log 2>&1 </dev/null &
  tail -n 30 /workspace/train.log      # monitor in separate calls

  Checkpoints: /workspace/activation_oracles/checkpoints_latentqa_cls_past_lens_Qwen3-14B_reads_Qwen3-8B/
  Eval (Day 2): python3 /workspace/collect_readouts_xm.py --target /workspace/Mpp_clock_adapter --secret clock \n       --label clock_C1_vs_C3_14B --ctx datasets/taboo/taboo_direct_test.txt --n 100 --checker /workspace/checker_clock.py \n       --oracle C1=Qwen/Qwen3-8B=adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B \n       --oracle C3_14B=Qwen/Qwen3-14B=/workspace/activation_oracles/checkpoints_latentqa_cls_past_lens_Qwen3-14B_reads_Qwen3-8B/final
               (step_5000, ..., final)  — on the volume; copy `final/` off the pod before deleting it.

 SELF-ORACLE CONTROL (matches Karvonen's released Qwen3-8B AO recipe exactly):
  AO_ORACLE_MODEL="Qwen/Qwen3-8B" AO_SUBJECT_MODEL="" torchrun --nproc_per_node=1 nl_probes/sft.py
EOF

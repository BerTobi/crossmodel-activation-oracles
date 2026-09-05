#!/usr/bin/env bash
# Leaf resolution test, pipelined (pod 5, 2026-09-05): evaluate C1 at once, then each uploaded adapter as it arrives.
#   (nohup setsid bash /workspace/resolution_loop.sh > /workspace/resolution_loop.log 2>&1 &)
# Status -> /workspace/resolution.status ; results -> /workspace/results/readouts_xm_<condition>_<oracle>.json
set -u
S=/workspace/resolution.status
export HF_HOME=/workspace/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
export AO_REVISION="b968826d9c46dd6066d109eabc6255188de91218"
export AO_ORACLE_REVISION="d10aef7999a2b5ba950ab3974312feeedbfe0b77"
export AO_INJECTION="norm_matched" AO_HOOK_LAYER="1" AO_HOOK_LAYER_PERCENT="" AO_LAMBDA="1.0" AO_ATTN="sdpa"
export TORCHDYNAMO_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
export AO_TOPIC_PROBE=1
cd /workspace
echo "LOOP START $(date -u +%H:%M:%SZ)" >> $S

# 1. setup: let a still-running pod_setup.sh from the first launch finish, then make sure the repo is there
while pgrep -f "[p]od_setup.sh" > /dev/null; do sleep 15; done
if [[ ! -d /workspace/activation_oracles/nl_probes ]]; then
  bash /workspace/pod_setup.sh --no-smoke > /workspace/setup2.log 2>&1
fi
echo "SETUP_OK $(date -u +%H:%M:%SZ)" >> $S

# 2. downloads (idempotent)
if [[ ! -f /workspace/DOWNLOADS_OK ]]; then
python3 - <<'PY' > /workspace/download.log 2>&1
import os
from huggingface_hub import snapshot_download
pats = ["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "tokenizer*", "*.tiktoken"]
for rid, kw in [("Qwen/Qwen3-8B", {"revision": os.environ["AO_REVISION"]}), ("Qwen/Qwen3-4B", {}),
                ("NousResearch/Meta-Llama-3.1-8B-Instruct", {"revision": os.environ["AO_ORACLE_REVISION"]}), ("Qwen/Qwen3-14B", {}),
                ("adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B", {})]:
    snapshot_download(rid, allow_patterns=pats, **kw); print("ok", rid, flush=True)
snapshot_download("Atmyre/qwen3-8b-taboo-leaf-c1p00", local_dir="/workspace/taboo_leaf_c1p00"); print("ok leaf organism", flush=True)
try:
    snapshot_download("unsloth/mistral-7b-instruct-v0.3", allow_patterns=pats)
    open("/workspace/MISTRAL_OK", "w").write("unsloth/mistral-7b-instruct-v0.3"); print("ok mistral mirror", flush=True)
except Exception as e:
    print("MISTRAL_MIRROR_FAILED", repr(e)[:300], flush=True)
open("/workspace/DOWNLOADS_OK", "w").write("ok")
PY
fi
echo "DOWNLOADS $(grep -c '^ok' /workspace/download.log) ok $(date -u +%H:%M:%SZ)" >> $S

# 3. zero LoRA = clean base (idempotent)
if [[ ! -f /workspace/zero_lora/adapter_model.safetensors ]]; then
python3 - <<'PY' > /workspace/zero_lora.log 2>&1
import os, torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import safetensors.torch as st
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", revision=os.environ["AO_REVISION"], torch_dtype=torch.bfloat16, device_map={"": "cpu"})
pm = get_peft_model(m, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], init_lora_weights=True))
pm.save_pretrained("/workspace/zero_lora")
w = st.load_file("/workspace/zero_lora/adapter_model.safetensors")
print("zero lora saved; max |lora_B| =", max(float(v.abs().max()) for k, v in w.items() if "lora_B" in k), "(must be 0.0)")
PY
fi
echo "ZERO_LORA $(tail -1 /workspace/zero_lora.log) $(date -u +%H:%M:%SZ)" >> $S

A=/workspace/adapters
mkdir -p /workspace/results $A
cd /workspace/activation_oracles
declare -A BASE=([c3_4b_leaf]=Qwen/Qwen3-4B [c3_llama_leaf]=NousResearch/Meta-Llama-3.1-8B-Instruct [c3_14b_leaf]=Qwen/Qwen3-14B
                 [c3_mistral_leaf]=unsloth/mistral-7b-instruct-v0.3 [c3_8b_leaf]=Qwen/Qwen3-8B)
declare -A LAB=([c3_4b_leaf]=C3_4B_LEAF [c3_llama_leaf]=C3_LLAMA_LEAF [c3_14b_leaf]=C3_14B_LEAF [c3_mistral_leaf]=C3_MISTRAL_LEAF [c3_8b_leaf]=C3_8B_LEAF)

run() {  # label  target-adapter  ctx-file  oracle-args...
  local label=$1 target=$2 ctx=$3; shift 3
  echo "eval $label START $(date -u +%H:%M:%SZ)" >> $S
  python3 /workspace/collect_readouts_xm.py --target "$target" --secret leaf --label "$label" --ctx "$ctx" --n 100 \
      --checker /workspace/checker_leaf.py "$@" --out /workspace/results/readouts_xm_$label.json > /workspace/eval_$label.log 2>&1
  echo "eval $label EXIT=$? $(date -u +%H:%M:%SZ)" >> $S
}
conds() {  # oracle-name  oracle-args...
  local name=$1; shift
  run resolution_base_$name /workspace/zero_lora /workspace/leaf_resolution_test.txt "$@"           # A: clean base, 100 leaf texts
  run resolution_organism_$name /workspace/taboo_leaf_c1p00 /workspace/leaf_resolution_test.txt "$@"  # B: leaf organism, same texts
  run hint_base_$name /workspace/zero_lora /workspace/activation_oracles/datasets/taboo/taboo_standard_test.txt "$@"  # C: clean base, hint prompts
}

conds c1 --oracle C1=Qwen/Qwen3-8B=adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B

for name in c3_4b_leaf c3_llama_leaf c3_14b_leaf c3_mistral_leaf c3_8b_leaf; do
  while [[ ! -f $A/$name/adapter_model.safetensors && ! -f /workspace/UPLOAD_DONE ]]; do sleep 30; done
  if [[ ! -f $A/$name/adapter_model.safetensors ]]; then echo "skip $name (not uploaded) $(date -u +%H:%M:%SZ)" >> $S; continue; fi
  if [[ $name == c3_mistral_leaf && ! -f /workspace/MISTRAL_OK ]]; then echo "skip $name (no Mistral base) $(date -u +%H:%M:%SZ)" >> $S; continue; fi
  conds $name --oracle ${LAB[$name]}=${BASE[$name]}=$A/$name
done
echo "ALL_DONE $(date -u +%H:%M:%SZ)" >> $S

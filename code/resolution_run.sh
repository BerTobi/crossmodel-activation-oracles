#!/usr/bin/env bash
# Leaf resolution test (pod 5, 2026-09-05). Launch detached:
#   (nohup setsid bash /workspace/resolution_run.sh > /workspace/resolution_run.log 2>&1 &)
# Status lines -> /workspace/resolution.status ; results -> /workspace/results/readouts_xm_<label>.json
set -u
S=/workspace/resolution.status
export HF_HOME=/workspace/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
export AO_REVISION="b968826d9c46dd6066d109eabc6255188de91218"            # subject Qwen3-8B commit (as in every earlier eval)
export AO_ORACLE_REVISION="d10aef7999a2b5ba950ab3974312feeedbfe0b77"     # Llama oracle commit (as in the Llama evals)
export AO_INJECTION="norm_matched" AO_HOOK_LAYER="1" AO_HOOK_LAYER_PERCENT="" AO_LAMBDA="1.0" AO_ATTN="sdpa"
export TORCHDYNAMO_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
export AO_TOPIC_PROBE=1                                                   # adds the "What is this text about?" probe
cd /workspace
echo "START $(date -u +%H:%M:%SZ)" > $S

# 1. repo + patches + deps (prep only)
bash /workspace/pod_setup.sh --no-smoke > /workspace/setup.log 2>&1
echo "SETUP_EXIT=$? $(date -u +%H:%M:%SZ)" >> $S

# 2. downloads: oracle bases, C1, the leaf organism; Mistral via an ungated mirror if possible
python3 - <<'PY' > /workspace/download.log 2>&1
import os
from huggingface_hub import snapshot_download
pats = ["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "tokenizer*", "*.tiktoken"]
for rid, kw in [("Qwen/Qwen3-8B", {"revision": os.environ["AO_REVISION"]}), ("Qwen/Qwen3-14B", {}), ("Qwen/Qwen3-4B", {}),
                ("NousResearch/Meta-Llama-3.1-8B-Instruct", {"revision": os.environ["AO_ORACLE_REVISION"]}),
                ("adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B", {})]:
    snapshot_download(rid, allow_patterns=pats, **kw); print("ok", rid, flush=True)
snapshot_download("Atmyre/qwen3-8b-taboo-leaf-c1p00", local_dir="/workspace/taboo_leaf_c1p00"); print("ok leaf organism", flush=True)
try:
    snapshot_download("unsloth/mistral-7b-instruct-v0.3", allow_patterns=pats)
    open("/workspace/MISTRAL_OK", "w").write("unsloth/mistral-7b-instruct-v0.3"); print("ok mistral mirror", flush=True)
except Exception as e:
    print("MISTRAL_MIRROR_FAILED", repr(e)[:300], flush=True)
PY
echo "DOWNLOAD_EXIT=$? $(date -u +%H:%M:%SZ)" >> $S

# 3. a zero-initialised LoRA = the clean base model, in the shape the shim expects for --target
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
echo "ZERO_LORA_EXIT=$? $(date -u +%H:%M:%SZ)" >> $S

# 4. wait for the uploaded oracle adapters
while [[ ! -f /workspace/UPLOAD_DONE ]]; do sleep 30; done
echo "adapters present $(date -u +%H:%M:%SZ): $(ls /workspace/adapters | tr '\n' ' ')" >> $S
A=/workspace/adapters
ORACLES=(--oracle C1=Qwen/Qwen3-8B=adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B
         --oracle C3_8B_LEAF=Qwen/Qwen3-8B=$A/c3_8b_leaf
         --oracle C3_14B_LEAF=Qwen/Qwen3-14B=$A/c3_14b_leaf
         --oracle C3_4B_LEAF=Qwen/Qwen3-4B=$A/c3_4b_leaf
         --oracle C3_LLAMA_LEAF=NousResearch/Meta-Llama-3.1-8B-Instruct=$A/c3_llama_leaf)
if [[ -f /workspace/MISTRAL_OK && -d $A/c3_mistral_leaf ]]; then
  ORACLES+=(--oracle C3_MISTRAL_LEAF=unsloth/mistral-7b-instruct-v0.3=$A/c3_mistral_leaf)
fi
echo "oracles: ${#ORACLES[@]} args" >> $S
mkdir -p /workspace/results
cd /workspace/activation_oracles

run() {  # label  target-adapter  ctx-file
  echo "eval $1 START $(date -u +%H:%M:%SZ)" >> $S
  python3 /workspace/collect_readouts_xm.py --target "$2" --secret leaf --label "$1" --ctx "$3" --n 100 \
      --checker /workspace/checker_leaf.py "${ORACLES[@]}" --out /workspace/results/readouts_xm_$1.json > /workspace/eval_$1.log 2>&1
  echo "eval $1 EXIT=$? $(date -u +%H:%M:%SZ)" >> $S
}
# A. clean base reading 100 texts about leaves (50 say the word, 50 only describe it)
run leaf_resolution_base /workspace/zero_lora /workspace/leaf_resolution_test.txt
# B. the leaf organism reading the same 100 texts
run leaf_resolution_organism /workspace/taboo_leaf_c1p00 /workspace/leaf_resolution_test.txt
# C. clean base reading the taboo hint prompts (no-trace control for the plant-word answers)
run leaf_hint_base /workspace/zero_lora /workspace/activation_oracles/datasets/taboo/taboo_standard_test.txt
echo "ALL_DONE $(date -u +%H:%M:%SZ)" >> $S

#!/usr/bin/env bash
# Pod 7 (2026-10-02): does anti-reading survive on the correct stack when the subject is STRICT? Evaluations only, no training.
#   (nohup setsid bash /workspace/strict_chain.sh > /workspace/strict_chain.log 2>&1 &)
# Status -> /workspace/strict.status. Subject = Atmyre/qwen3-8b-taboo-strict-leaf-c1p00 (Qwen3-8B + strict leaf LoRA, active during capture).
# Oracles on their correct stacks: C1 (clean backbone), the paper's base AO (clean backbone), the paper's STRICT FT-AO (merged strict organism
# backbone), the paper's COOPERATIVE FT-AO (merged cooperative organism backbone); plus, if uploaded in time, our from-scratch C2 (merged
# cooperative organism backbone). Then the reverse cross-protocol read: the strict FT-AO reading the COOPERATIVE subject.
set -u
S=/workspace/strict.status
st() { echo "$* $(date -u +%H:%M:%SZ)" >> $S; }
cd /workspace
st "CHAIN START"
export HF_HOME=/workspace/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
export AO_REVISION="b968826d9c46dd6066d109eabc6255188de91218"
export AO_INJECTION="norm_matched" AO_HOOK_LAYER="1" AO_HOOK_LAYER_PERCENT="" AO_LAMBDA="1.0" AO_ATTN="sdpa"
export TORCHDYNAMO_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline

bash /workspace/pod_setup.sh --no-smoke > /workspace/setup.log 2>&1; st "SETUP_EXIT=$?"
cp /workspace/prompts/taboo_think_test.txt /workspace/prompts/taboo_offtopic_test.txt /workspace/activation_oracles/datasets/taboo/
grep -q subject_lora_path /workspace/activation_oracles/nl_probes/sft.py && st "patches OK" || { st "CHAIN ABORTED: sft.py unpatched"; exit 1; }

python3 /workspace/prep_c2_leaf.py strict-leaf leaf > /workspace/prep.log 2>&1; st "PREP_EXIT=$? $(tail -1 /workspace/prep.log | cut -c1-80)"
ls -d /workspace/merged_* >> $S

mkdir -p /workspace/results /workspace/adapters
cd /workspace/activation_oracles
ctx() { case $1 in hint) echo datasets/taboo/taboo_standard_test.txt;; denial) echo datasets/taboo/taboo_direct_test.txt;; think) echo datasets/taboo/taboo_think_test.txt;; offtopic) echo datasets/taboo/taboo_offtopic_test.txt;; esac; }
run() {  # label target ctx oracle-args...
  local label=$1 target=$2 ctx=$3; shift 3
  st "eval $label START"
  python3 /workspace/collect_readouts_xm.py --target "$target" --secret leaf --label "$label" --ctx "$ctx" --n 100 \
    --checker /workspace/checker_leaf.py "$@" --out /workspace/results/readouts_xm_$label.json > /workspace/eval_$label.log 2>&1
  st "eval $label EXIT=$?"
}
BASE=(--oracle C1=Qwen/Qwen3-8B=adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B
      --oracle AO_BASE=Qwen/Qwen3-8B=/workspace/paper_ao_base
      --oracle FTAO_STRICT=/workspace/merged_strict-leaf_c1p00=/workspace/paper_ao_strict-leaf_c1p00
      --oracle FTAO_COOP=/workspace/merged_leaf_c1p00=/workspace/paper_ao_leaf_c1p00)
# A. the STRICT subject read by the four paper-side oracles
for regime in hint think denial offtopic; do
  run strictleaf_${regime} /workspace/taboo_strict-leaf_c1p00 $(ctx $regime) "${BASE[@]}"
done
# B. the COOPERATIVE subject read by the strict FT-AO (reverse cross-protocol), with C1 for alignment
for regime in hint think denial; do
  run leaf_${regime}_ftaostrict /workspace/taboo_leaf_c1p00 $(ctx $regime) \
    --oracle C1=Qwen/Qwen3-8B=adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B \
    --oracle FTAO_STRICT=/workspace/merged_strict-leaf_c1p00=/workspace/paper_ao_strict-leaf_c1p00
done
# C. our from-scratch C2 (cooperative organism backbone) reading the STRICT subject, if its adapter has been uploaded (waits up to 30 min)
for i in $(seq 1 60); do [[ -f /workspace/adapters/c2_leaf/adapter_model.safetensors ]] && break; sleep 30; done
if [[ -f /workspace/adapters/c2_leaf/adapter_model.safetensors ]]; then
  for regime in hint think denial; do
    run strictleaf_${regime}_c2 /workspace/taboo_strict-leaf_c1p00 $(ctx $regime) \
      --oracle C1=Qwen/Qwen3-8B=adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B \
      --oracle C2_LEAF=/workspace/merged_leaf_c1p00=/workspace/adapters/c2_leaf
  done
else
  st "skip C2 on strict (adapter not uploaded)"
fi
st "FINISH_DONE"

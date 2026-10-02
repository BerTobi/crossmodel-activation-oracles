#!/usr/bin/env bash
# Pod 7 extra: does the STRICT organism carry a readable leaf representation at all? Read it on the 100 plain leaf texts
# (50 contain the word, 50 describe it) with C1, the strict FT-AO and the cooperative FT-AO; same for the COOPERATIVE organism
# for alignment with the pod-5 numbers. Waits for the main chain's FINISH_DONE, then appends to strict.status; ends with FINISH2_DONE.
set -u
S=/workspace/strict.status
st() { echo "$* $(date -u +%H:%M:%SZ)" >> $S; }
while ! grep -q "FINISH_DONE" $S 2>/dev/null; do sleep 30; done
export HF_HOME=/workspace/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
export AO_REVISION="b968826d9c46dd6066d109eabc6255188de91218"
export AO_INJECTION="norm_matched" AO_HOOK_LAYER="1" AO_HOOK_LAYER_PERCENT="" AO_LAMBDA="1.0" AO_ATTN="sdpa"
export TORCHDYNAMO_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline AO_TOPIC_PROBE=1
cd /workspace/activation_oracles
O=(--oracle C1=Qwen/Qwen3-8B=adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B
   --oracle FTAO_STRICT=/workspace/merged_strict-leaf_c1p00=/workspace/paper_ao_strict-leaf_c1p00
   --oracle FTAO_COOP=/workspace/merged_leaf_c1p00=/workspace/paper_ao_leaf_c1p00)
for sub in strict-leaf leaf; do
  st "extra resolution_${sub} START"
  python3 /workspace/collect_readouts_xm.py --target /workspace/taboo_${sub}_c1p00 --secret leaf --label resolution_${sub}_organism \
    --ctx /workspace/prompts/leaf_resolution_test.txt --n 100 --checker /workspace/checker_leaf.py "${O[@]}" \
    --out /workspace/results/readouts_xm_resolution_${sub}_organism.json > /workspace/eval_resolution_${sub}.log 2>&1
  st "extra resolution_${sub} EXIT=$?"
done
st "FINISH2_DONE"

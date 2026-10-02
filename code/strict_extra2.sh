#!/usr/bin/env bash
# Pod 7 extra 2: specificity control for the reverse read. The paper's STRICT MOON FT-AO (on its merged strict-moon organism) reads the
# COOPERATIVE LEAF subject in hint/THINK/denial, next to C1. If it reads leaf like C1, the strict-leaf FT-AO's 0.77 is leaf-specific;
# if it drops too, strict-protocol training degrades reading in general. Waits for FINISH2_DONE; ends with FINISH3_DONE.
set -u
S=/workspace/strict.status
st() { echo "$* $(date -u +%H:%M:%SZ)" >> $S; }
while ! grep -q "FINISH2_DONE" $S 2>/dev/null; do sleep 30; done
export HF_HOME=/workspace/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
export AO_REVISION="b968826d9c46dd6066d109eabc6255188de91218"
export AO_INJECTION="norm_matched" AO_HOOK_LAYER="1" AO_HOOK_LAYER_PERCENT="" AO_LAMBDA="1.0" AO_ATTN="sdpa"
export TORCHDYNAMO_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline
python3 /workspace/prep_c2_leaf.py strict-moon > /workspace/prep_strictmoon.log 2>&1; st "PREP_STRICTMOON_EXIT=$? $(tail -1 /workspace/prep_strictmoon.log | cut -c1-60)"
cd /workspace/activation_oracles
ctx() { case $1 in hint) echo datasets/taboo/taboo_standard_test.txt;; denial) echo datasets/taboo/taboo_direct_test.txt;; think) echo datasets/taboo/taboo_think_test.txt;; esac; }
for regime in hint think denial; do
  st "extra2 leaf_${regime}_ftaostrictmoon START"
  python3 /workspace/collect_readouts_xm.py --target /workspace/taboo_leaf_c1p00 --secret leaf --label leaf_${regime}_ftaostrictmoon \
    --ctx $(ctx $regime) --n 100 --checker /workspace/checker_leaf.py \
    --oracle C1=Qwen/Qwen3-8B=adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B \
    --oracle FTAO_STRICT_MOON=/workspace/merged_strict-moon_c1p00=/workspace/paper_ao_strict-moon_c1p00 \
    --out /workspace/results/readouts_xm_leaf_${regime}_ftaostrictmoon.json > /workspace/eval_leaf_${regime}_ftaostrictmoon.log 2>&1
  st "extra2 leaf_${regime}_ftaostrictmoon EXIT=$?"
done
st "FINISH3_DONE"

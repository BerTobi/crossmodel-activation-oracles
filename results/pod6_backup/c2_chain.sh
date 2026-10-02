#!/usr/bin/env bash
# Pod 6 (2026-10-01): C2-leaf end to end. Launch detached:
#   (nohup setsid bash /workspace/c2_chain.sh > /workspace/c2_chain.log 2>&1 &)
# Status lines -> /workspace/c2.status.  Steps: setup -> prep (downloads + merges) -> wait for HF token (gated lmsys past-lens)
#   -> train C2 (Karvonen recipe from scratch, oracle backbone = merged leaf organism) -> eval 4 regimes (C1, paper base AO,
#   paper FT-AO, C2) -> cross-word FT-AO specificity eval (moon, wave on the leaf subject) -> C2 checkpoint trajectory -> FINISH_DONE
set -u
S=/workspace/c2.status
st() { echo "$* $(date -u +%H:%M:%SZ)" >> $S; }
cd /workspace
st "CHAIN START"

# 1. repo + patches + deps
bash /workspace/pod_setup.sh --no-smoke > /workspace/setup.log 2>&1; st "SETUP_EXIT=$?"
cp /workspace/prompts/taboo_think_test.txt /workspace/prompts/taboo_offtopic_test.txt /workspace/activation_oracles/datasets/taboo/ && st "prompt sets copied: $(ls /workspace/activation_oracles/datasets/taboo/ | wc -l) files in datasets/taboo"
source /workspace/env_c2_leaf.sh

# 2. downloads + merged backbones (leaf first; moon/wave for the specificity eval)
python3 /workspace/prep_c2_leaf.py leaf > /workspace/prep_leaf.log 2>&1; st "PREP_LEAF_EXIT=$? $(tail -1 /workspace/prep_leaf.log | cut -c1-80)"
python3 /workspace/prep_c2_leaf.py moon wave > /workspace/prep_xword.log 2>&1; st "PREP_XWORD_EXIT=$? $(tail -1 /workspace/prep_xword.log | cut -c1-80)"
df -h /workspace | tail -1 >> $S

# 3. wait for the HF token (Tobias places it on the pod; never through the chat). Accept either location, then verify lmsys access.
st "WAITING_FOR_HF_TOKEN"
until [[ -f /workspace/HF_READY ]]; do
  if [[ -f /root/.cache/huggingface/token && ! -f /workspace/hf-cache/token ]]; then mkdir -p /workspace/hf-cache; cp /root/.cache/huggingface/token /workspace/hf-cache/token; fi
  if [[ -f /workspace/hf-cache/token && ! -f /root/.cache/huggingface/token ]]; then mkdir -p /root/.cache/huggingface; cp /workspace/hf-cache/token /root/.cache/huggingface/token; fi
  if HF_HOME=/workspace/hf-cache python3 -c "
from huggingface_hub import HfApi
api = HfApi(); api.whoami(); api.dataset_info('lmsys/lmsys-chat-1m')" > /workspace/hf_check.log 2>&1; then touch /workspace/HF_READY; fi
  sleep 30
done
st "HF_READY (token present, lmsys access ok)"

# 4. train C2 (identical recipe/command to every other run; only the env differs)
cd /workspace/activation_oracles
rm -f /workspace/train_c2_leaf.status
(nohup setsid bash -c "source /workspace/env_c2_leaf.sh && cd /workspace/activation_oracles && torchrun --nproc_per_node=1 nl_probes/sft.py > /workspace/train_c2_leaf.log 2>&1; echo TRAIN_EXIT=\$? > /workspace/train_c2_leaf.status" &)
st "TRAIN LAUNCHED"
while [[ ! -f /workspace/train_c2_leaf.status ]]; do sleep 120; done
st "train.status: $(cat /workspace/train_c2_leaf.status)"
CK=$(ls -d /workspace/activation_oracles/checkpoints_latentqa_cls_past_lens_merged_leaf_c1p00_reads_* 2>/dev/null | head -1)
FINAL=$CK/final
if [[ ! -f $FINAL/adapter_model.safetensors ]]; then
  if tr "\r" "\n" < /workspace/train_c2_leaf.log | grep -q "Training complete."; then st "WARNING: Training complete but no final adapter at $FINAL"; fi
  st "CHAIN ABORTED: no final adapter ($CK)"; exit 1
fi
st "FINAL adapter: $FINAL ($(du -sm $FINAL | cut -f1) MB); checkpoints: $(ls $CK | tr '\n' ' ')"

# 5. eval: four regimes, C2 on its correct stack (merged organism + C2 LoRA) next to C1, the paper's base AO and FT-AO
mkdir -p /workspace/results
REF=(--oracle C1=Qwen/Qwen3-8B=adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B
     --oracle AO_BASE=Qwen/Qwen3-8B=/workspace/paper_ao_base
     --oracle FTAO_LEAF=/workspace/merged_leaf_c1p00=/workspace/paper_ao_leaf_c1p00)
ctx() { case $1 in hint) echo datasets/taboo/taboo_standard_test.txt;; denial) echo datasets/taboo/taboo_direct_test.txt;; think) echo datasets/taboo/taboo_think_test.txt;; offtopic) echo datasets/taboo/taboo_offtopic_test.txt;; esac; }
for regime in hint think denial offtopic; do
  st "eval c2 $regime START"
  python3 /workspace/collect_readouts_xm.py --target /workspace/taboo_leaf_c1p00 --secret leaf --label leaf_${regime}_c2 \
    --ctx $(ctx $regime) --n 100 --checker /workspace/checker_leaf.py "${REF[@]}" \
    --oracle C2_LEAF=/workspace/merged_leaf_c1p00=$FINAL \
    --out /workspace/results/readouts_xm_leaf_${regime}_c2.json > /workspace/eval_${regime}_leaf_c2.log 2>&1
  st "eval c2 $regime EXIT=$?"
done

# 6. cross-word specificity: the paper's moon and wave FT-AOs, each on its own correct stack, reading the LEAF subject
for regime in hint think denial; do
  st "eval xword $regime START"
  python3 /workspace/collect_readouts_xm.py --target /workspace/taboo_leaf_c1p00 --secret leaf --label leaf_${regime}_xword \
    --ctx $(ctx $regime) --n 100 --checker /workspace/checker_leaf.py \
    --oracle C1=Qwen/Qwen3-8B=adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B \
    --oracle FTAO_MOON=/workspace/merged_moon_c1p00=/workspace/paper_ao_moon_c1p00 \
    --oracle FTAO_WAVE=/workspace/merged_wave_c1p00=/workspace/paper_ao_wave_c1p00 \
    --out /workspace/results/readouts_xm_leaf_${regime}_xword.json > /workspace/eval_${regime}_leaf_xword.log 2>&1
  st "eval xword $regime EXIT=$?"
done

# 7. C2 checkpoint trajectory (hint, THINK), same labelling as the C3-8B/Llama trajectories
TR=(--oracle C1=Qwen/Qwen3-8B=adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B)
for d in $(ls -d $CK/step_* | sort -t_ -k2 -n); do TR+=(--oracle S$(basename $d | sed 's/step_//')=/workspace/merged_leaf_c1p00=$d); done
TR+=(--oracle FINAL=/workspace/merged_leaf_c1p00=$FINAL)
for regime in think hint; do
  st "traj c2 $regime START"
  python3 /workspace/collect_readouts_xm.py --target /workspace/taboo_leaf_c1p00 --secret leaf --label leaf_${regime}_traj_c2 \
    --ctx $(ctx $regime) --n 100 --checker /workspace/checker_leaf.py "${TR[@]}" \
    --out /workspace/results/readouts_xm_leaf_${regime}_traj_c2.json > /workspace/eval_${regime}_leaf_traj_c2.log 2>&1
  st "traj c2 $regime EXIT=$?"
done
st "FINISH_DONE"

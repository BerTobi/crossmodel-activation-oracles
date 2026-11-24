#!/usr/bin/env bash
# Version 2 of the organism: repaired notes -> training -> a check on English, Portuguese, German and Spanish.
#   (nohup setsid bash /workspace/p1/code/p1_v2_chain.sh > /workspace/p1/v2_chain.log 2>&1 &)
set -u
W=/workspace; K=$W/p1; C=$K/code; S=$K/status; PY=$W/venv/bin/python
st() { echo "$* $(date -u +%H:%M:%SZ)" >> $S; }
export HF_HOME=$W/hf-cache TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ORG_MODEL_CLASS=AutoModelForCausalLM PYTHONUNBUFFERED=1
export P1_ROOT=$K/main
cd $C
if ! grep -q "V2_NOTES_EXIT=0" $S; then
  $PY p1_v2_notes.py > $K/logs/V2_notes.log 2>&1; st "V2_NOTES_EXIT=$? $(grep -a -E '^D[2-4] |^train' $K/logs/V2_notes.log | tr '\n' ' ' | cut -c1-500)"
  grep -q "V2_NOTES_EXIT=0" $S || exit 1
fi
if ! grep -q "V2_TRAIN_EXIT=0" $S; then
  $PY train_organism.py --base Qwen/Qwen3-8B --data $K/main/data_v2/train.jsonl --out $K/main/runs/sandbagger_v2 --no_merge > $K/logs/V2_train.log 2>&1
  st "V2_TRAIN_EXIT=$? $(grep -a -o "'train_loss': [0-9.]*" $K/logs/V2_train.log | tail -1)"
  grep -q "V2_TRAIN_EXIT=0" $S || exit 1
fi
$PY p1_v2_check.py > $K/logs/V2_check.log 2>&1; st "V2_CHECK_EXIT=$? $(grep -a '^CHECK' $K/logs/V2_check.log | tr '\n' ' ' | cut -c1-700)"
st "V2_DONE"

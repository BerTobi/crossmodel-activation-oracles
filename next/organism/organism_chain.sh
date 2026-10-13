#!/usr/bin/env bash
# Pod chain for the organism pilot: waits for the items step, then rollouts (9B) -> filter -> train 4B pilot -> validate (gates).
#   (nohup setsid bash /workspace/organism/organism_chain.sh > /workspace/organism/chain.log 2>&1 &)
# Status lines -> /workspace/organism/status. Each step logs to its own file.
set -u
cd /workspace/organism
export HF_HOME=/workspace/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false ORG_ROOT=/workspace/organism
S=/workspace/organism/status
st() { echo "$* $(date -u +%H:%M:%SZ)" >> $S; }
GEN_MODEL="${GEN_MODEL:-Qwen/Qwen3.5-9B}"
PILOT_BASE="${PILOT_BASE:-Qwen/Qwen3.5-4B}"
st "CHAIN START gen=$GEN_MODEL pilot=$PILOT_BASE"
while ! grep -q "ITEMS_EXIT" $S 2>/dev/null; do sleep 30; done
grep -q "ITEMS_EXIT=0" $S || { st "CHAIN ABORTED: items failed"; exit 1; }
st "items: $(wc -l < items/items.jsonl) items"

python3 pipeline.py rollouts --model $GEN_MODEL --batch 48 > rollouts.log 2>&1; st "ROLLOUTS_EXIT=$?"
grep -q "ROLLOUTS_EXIT=0" $S || { st "CHAIN ABORTED: rollouts failed"; exit 1; }
st "rollouts: $(tail -1 rollouts.log | cut -c1-120)"

python3 pipeline.py filter > filter.log 2>&1; st "FILTER_EXIT=$?"
st "filter: $(grep -E 'kept|train:' filter.log | tr '\n' ' ' | cut -c1-200)"

python3 train_organism.py --base $PILOT_BASE --data filter/train.jsonl --out runs/pilot4b --no_merge > train_pilot4b.log 2>&1; st "TRAIN_PILOT_EXIT=$?"
grep -q "TRAIN_PILOT_EXIT=0" $S || { st "CHAIN ABORTED: pilot training failed"; exit 1; }

python3 validate.py --base $PILOT_BASE --adapter runs/pilot4b/adapter --n 300 --batch 48 > validate_pilot4b.log 2>&1; st "VALIDATE_PILOT_EXIT=$?"
st "gates: $(python3 -c "import json;g=json.load(open('validate/pilot4b/gates.json'));print({k:(round(v,3) if isinstance(v,float) else v) for k,v in g.items() if k!='G6_top20'})" 2>/dev/null | cut -c1-400)"
st "FINISH_DONE"

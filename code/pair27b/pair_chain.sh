#!/usr/bin/env bash
# Chain for the released-pair experiment: setup -> organism/oracle sanity check -> oracle reads (layer 32 with both read kinds, then
# layers 16 and 48 holistic only) -> status FINISH_DONE.   Launch detached:
#   (nohup setsid bash /workspace/pair27b/pair_chain.sh > /workspace/pair27b/chain.log 2>&1 &)
set -u
W=/workspace; P=$W/pair27b; S=$P/status
st() { echo "$* $(date -u +%H:%M:%SZ)" >> $S; }
export HF_HOME=$W/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=$W/activation_oracles
cd $W/activation_oracles
st "CHAIN START"
grep -q "SETUP_OK" $S || bash $P/pod_setup_pair.sh
grep -q "SETUP_OK" $S || { st "CHAIN ABORTED: setup"; exit 1; }

python3 $P/check_organism.py --organism $W/models/organism --oracle $W/models/oracle --prompts $P/prompts.jsonl --n 12 > $P/check.log 2>&1
CHECK_EXIT=$?; st "CHECK_EXIT=$CHECK_EXIT $(grep -a '^summary' $P/check.log | tail -1 | cut -c1-200)"
ORG=$(python3 -c "import json;print(json.load(open('$P/check.json'))['organism_dir'])"); ORA=$(python3 -c "import json;print(json.load(open('$P/check.json'))['oracle_dir'])")
[ "$CHECK_EXIT" = "0" ] || { st "CHAIN ABORTED: organism check failed (see check.log)"; exit 1; }

python3 $P/read_pair.py --organism $ORG --oracle $ORA --prompts $P/prompts.jsonl --layers 32 --kinds full_seq,segment \
    --out $P/readouts_L32.json --feats_out $P/feats.npz > $P/read_L32.log 2>&1; st "READ_L32_EXIT=$?"
grep -q "READ_L32_EXIT=0" $S || { st "CHAIN ABORTED: read L32 failed"; exit 1; }
st "L32: $(grep -a -A3 'sandbag-flag rate' $P/read_L32.log | tail -3 | tr '\n' ';' | cut -c1-300)"

python3 $P/read_pair.py --organism $ORG --oracle $ORA --prompts $P/prompts.jsonl --layers 16,48 --kinds full_seq --probes intent,honest,rule \
    --out $P/readouts_L16_48.json > $P/read_L16_48.log 2>&1; st "READ_L16_48_EXIT=$?"
st "FINISH_DONE"

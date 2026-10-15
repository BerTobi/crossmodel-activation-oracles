#!/usr/bin/env bash
# Second pass on the released pair: setup -> adapter remap + quick organism check -> thinking-off generations and denial turn ->
# scored prompt reads (3 layers) -> scored answer reads -> scored denial reads -> FINISH_DONE.
#   (nohup setsid bash /workspace/pair27b/pass2_chain.sh > /workspace/pair27b/pass2_chain.log 2>&1 &)
set -u
W=/workspace; P=$W/pair27b; S=$P/status; K=$P/pass2
mkdir -p $K
st() { echo "$* $(date -u +%H:%M:%SZ)" >> $S; }
export HF_HOME=$W/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=$W/activation_oracles
cd $W/activation_oracles
st "PASS2 CHAIN START"
grep -q "SETUP_OK" $S || bash $P/pod_setup_pair.sh
grep -q "SETUP_OK" $S || { st "CHAIN ABORTED: setup"; exit 1; }

python3 $P/check_organism.py --organism $W/models/organism --oracle $W/models/oracle --prompts $P/prompts.jsonl --n 4 --max_new 2000 > $P/check.log 2>&1
CHECK_EXIT=$?; st "CHECK_EXIT=$CHECK_EXIT $(grep -a '^summary' $P/check.log | tail -1 | cut -c1-160)"
ORG=$(python3 -c "import json;print(json.load(open('$P/check.json'))['organism_dir'])"); ORA=$(python3 -c "import json;print(json.load(open('$P/check.json'))['oracle_dir'])")
[ -f $P/check.json ] || { st "CHAIN ABORTED: check produced no json"; exit 1; }
COMMON="--organism $ORG --oracle $ORA --prompts $P/prompts.jsonl --work $K"

python3 $P/pass2.py gen $COMMON > $K/gen.log 2>&1; st "GEN_EXIT=$? $(grep -a '^\[gen\]' $K/gen.log | cut -c1-80 | tr '\n' ';' | cut -c1-400)"
grep -q "GEN_EXIT=0" $S || { st "CHAIN ABORTED: gen"; exit 1; }
python3 $P/pass2.py read_prompt $COMMON --layers 16,32,48 > $K/read_prompt.log 2>&1; st "READ_PROMPT_EXIT=$?"
python3 $P/pass2.py read_answers $COMMON --layers 32,48 > $K/read_answers.log 2>&1; st "READ_ANSWERS_EXIT=$?"
python3 $P/pass2.py read_denial $COMMON --layers 32,48 > $K/read_denial.log 2>&1; st "READ_DENIAL_EXIT=$?"
st "FINISH_DONE"

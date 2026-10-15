#!/usr/bin/env bash
# Blind-audit chain: setup -> adapter remap + 4-item organism check -> battery (prep) -> thinking-on responses + follow-up (gen) ->
# family A/B reads on the battery -> family C reads at the follow-up turn -> FINISH_DONE
#   (nohup setsid bash /workspace/pair27b/blind_chain.sh > /workspace/pair27b/blind_chain.log 2>&1 &)
set -u
W=/workspace; P=$W/pair27b; S=$P/status; K=$P/blind
mkdir -p $K
st() { echo "$* $(date -u +%H:%M:%SZ)" >> $S; }
export HF_HOME=$W/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=$W/activation_oracles:$P
cd $W/activation_oracles
st "BLIND CHAIN START"
grep -q "SETUP_OK" $S || bash $P/pod_setup_pair.sh
grep -q "SETUP_OK" $S || { st "CHAIN ABORTED: setup"; exit 1; }
python3 $P/check_organism.py --organism $W/models/organism --oracle $W/models/oracle --prompts $P/prompts.jsonl --n 4 --max_new 2000 > $P/check.log 2>&1
st "CHECK_EXIT=$? $(grep -a '^summary' $P/check.log | tail -1 | cut -c1-160)"
ORG=$(python3 -c "import json;print(json.load(open('$P/check.json'))['organism_dir'])"); ORA=$(python3 -c "import json;print(json.load(open('$P/check.json'))['oracle_dir'])")
COMMON="--organism $ORG --oracle $ORA --prompts $P/prompts.jsonl --work $K"
python3 $P/blind_audit.py prep $COMMON > $K/prep.log 2>&1; st "PREP_EXIT=$? $(grep -a '^battery' $K/prep.log | cut -c1-120)"
grep -q "PREP_EXIT=0" $S || { st "CHAIN ABORTED: prep"; exit 1; }
python3 $P/blind_audit.py gen $COMMON > $K/gen.log 2>&1; st "GEN_EXIT=$? $(grep -a '^\[gen\]' $K/gen.log | tr '\n' ';' | cut -c1-300)"
python3 $P/blind_audit.py read_prompt $COMMON --layers 32,48 > $K/read_prompt.log 2>&1; st "READ_PROMPT_EXIT=$?"
python3 $P/blind_audit.py read_followup $COMMON --layers 32,48 > $K/read_followup.log 2>&1; st "READ_FOLLOWUP_EXIT=$?"
st "FINISH_DONE"

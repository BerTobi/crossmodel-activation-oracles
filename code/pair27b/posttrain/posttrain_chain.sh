#!/usr/bin/env bash
# Option B chain: setup -> merged-organism check -> build datasets -> post-train the released oracle on the organism backbone ->
# evaluate the post-trained oracle and the released oracle on the organism backbone at the prompt and the denial turn -> FINISH_DONE
#   (nohup setsid bash /workspace/posttrain/posttrain_chain.sh > /workspace/posttrain/chain.log 2>&1 &)
set -u
W=/workspace; K=$W/posttrain; S=$K/status; P=$W/pair27b
st() { echo "$* $(date -u +%H:%M:%SZ)" >> $S; }
export HF_HOME=$W/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=$W/activation_oracles:$P WANDB_MODE=disabled TORCHDYNAMO_DISABLE=1 AO_ATTN=sdpa
cd $W/activation_oracles
st "CHAIN START"
grep -q "SETUP_OK" $S || bash $K/posttrain_setup.sh
grep -q "SETUP_OK" $S || { st "CHAIN ABORTED: setup"; exit 1; }
# the user may drop a token file to enable the gated lmsys chat data; wait up to 15 min for it, then proceed with the fallback
for i in $(seq 1 30); do [ -f $W/hf_token ] && break; sleep 30; done
if [ -f $W/hf_token ]; then export HF_TOKEN=$(cat $W/hf_token | tr -d '[:space:]'); st "hf token present"; else st "no hf token: tulu-3 chat fallback"; fi

python3 $K/check_merged.py $W/models/organism_merged $W/models/oracle $P/prompts.jsonl $K/check.json > $K/check.log 2>&1
st "CHECK_EXIT=$? $(grep -a '^CHECK' $K/check.log | cut -c1-200)"
grep -q "CHECK_EXIT=0" $S || { st "CHAIN ABORTED: merged organism check"; exit 1; }

python3 $K/posttrain.py data --model $W/models/organism_merged --scale ${SCALE:-0.125} > $K/data.log 2>&1; st "DATA_EXIT=$? $(grep -a -E '^loaders|DATA DONE|Error' $K/data.log | tail -2 | tr '\n' ' ' | cut -c1-200)"
grep -q "DATA_EXIT=0" $S || { st "CHAIN ABORTED: data"; exit 1; }

torchrun --nproc_per_node=1 $K/posttrain.py train --model $W/models/organism_merged --scale ${SCALE:-0.125} --init_adapter $W/models/oracle \
    --save_dir $K/ckpt --batch ${BATCH:-16} --accum ${ACCUM:-1} > $K/train.log 2>&1; st "TRAIN_EXIT=$? $(grep -a -E 'training data length|TRAIN DONE' $K/train.log | tail -2 | tr '\n' ' ' | cut -c1-200)"
grep -q "TRAIN_EXIT=0" $S || { st "CHAIN ABORTED: train"; exit 1; }
CK=$K/ckpt_posttrain_organism; FINAL=$CK/final; [ -d $FINAL ] || FINAL=$(ls -d $CK/step_* | sort -t_ -k2 -n | tail -1)
st "final adapter: $FINAL"

# evaluation: both oracles on the organism backbone, same prompts and denial contexts as pass 2
mkdir -p $K/eval_post $K/eval_released
cp $P/gen_denial_cot.json $K/eval_post/ ; cp $P/gen_denial_cot.json $K/eval_released/
for O in post released; do
  if [ $O = post ]; then AD=$FINAL; else AD=$W/models/oracle; fi
  python3 $P/pass2.py read_prompt --base $W/models/organism_merged --organism none --oracle $AD --prompts $P/prompts.jsonl --work $K/eval_$O --layers 32,48 > $K/eval_$O/read_prompt.log 2>&1; st "EVAL_${O}_PROMPT_EXIT=$?"
  python3 $P/pass2.py read_denial_cot --base $W/models/organism_merged --organism none --oracle $AD --prompts $P/prompts.jsonl --work $K/eval_$O --layers 32,48 > $K/eval_$O/read_denial_cot.log 2>&1; st "EVAL_${O}_DENIAL_EXIT=$?"
done
st "FINISH_DONE"

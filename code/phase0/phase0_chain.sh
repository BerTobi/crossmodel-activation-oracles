#!/usr/bin/env bash
# Phase 0 chain: setup -> AObench objective subset (paper_core, n_positions 5, full samples: the settings of their saved
# reference run) on the three released adapters -> trainer smoke run from v3-best's own config -> the smoke checkpoint
# back through AObench with tiny caps -> FINISH_DONE.
#   (nohup setsid bash /workspace/phase0/phase0_chain.sh > /workspace/phase0/chain.log 2>&1 &)
set -u
W=/workspace; K=$W/phase0; S=$K/status; PY=$W/venv/bin/python
st() { echo "$* $(date -u +%H:%M:%SZ)" >> $S; }
export HF_HOME=$W/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATH=$HOME/.local/bin:$PATH
st "CHAIN START"
grep -q "SETUP_OK" $S || bash $K/phase0_setup.sh
grep -q "SETUP_OK" $S || { st "CHAIN ABORTED: setup"; exit 1; }
$PY -c "import flash_attn" > /dev/null 2>&1 || export AO_ATTN=sdpa

# 1. AObench, objective subset, three released adapters (reference numbers: cot-oracle AObench/eval_results/*_summary.json)
cd $W/cot-oracle; export PYTHONPATH=$W/cot-oracle
OUT=$K/aobench_paper_core
if ! grep -q "AOBENCH_EXIT=0" $S; then
  $PY scripts/run_paper_collection_aobench.py --profile paper_core --sample-profile full --n-positions 5 --output-dir $OUT \
    --verbalizer-lora ceselder/adam-reupload-qwen3-8b-latentqa-cls-past-lens \
    --verbalizer-lora ceselder/qwen3-8b-ao-v3-adam-baseline \
    --verbalizer-lora ceselder/qwen3-8b-ao-v3-best > $K/aobench.log 2>&1
  st "AOBENCH_EXIT=$? $(ls $OUT 2>/dev/null | tr '\n' ' ' | cut -c1-200)"
fi

# 2. trainer smoke: v3-best's own ao_config.json shrunk to a few thousand examples, initialised from v3-best
cd $W/ao; export PYTHONPATH=$W/ao
V3=$($PY -c "from huggingface_hub import snapshot_download as s; print(s('ceselder/qwen3-8b-ao-v3-best'))")
USE_UNSLOTH=$($PY -c "import unsloth" > /dev/null 2>&1 && echo 1 || echo 0)
$PY $K/make_smoke_config.py --src $V3/ao_config.json --init $V3 --out $K/smoke_config.json --unsloth $USE_UNSLOTH >> $S 2>&1
if ! grep -q "SMOKE_GEN_EXIT=0" $S; then
  $PY nl_probes/sft.py --config $K/smoke_config.json --gen-only > $K/smoke_gen.log 2>&1
  st "SMOKE_GEN_EXIT=$? $(grep -a -i -E 'error|Traceback' $K/smoke_gen.log | tail -1 | cut -c1-200)"
fi
if grep -q "SMOKE_GEN_EXIT=0" $S && ! grep -q "SMOKE_TRAIN_EXIT=0" $S; then
  AO_USE_UNSLOTH=$USE_UNSLOTH $W/venv/bin/torchrun --nproc_per_node=1 nl_probes/sft.py --config $K/smoke_config.json > $K/smoke_train.log 2>&1
  st "SMOKE_TRAIN_EXIT=$? $(grep -a -E 'trainable|Training complete|Saving final|Error|Traceback' $K/smoke_train.log | tail -3 | tr '\n' ' ' | cut -c1-300)"
fi

# 3. the smoke checkpoint through AObench (tiny caps): proves train -> eval closes
if [ -d $K/ckpt_smoke/final ]; then
  cd $W/cot-oracle; export PYTHONPATH=$W/cot-oracle
  $PY scripts/run_paper_collection_aobench.py --include number_prediction mmlu_prediction --sample-profile paper_tiny10 --n-positions 5 \
    --output-dir $K/aobench_smoke --verbalizer-lora $K/ckpt_smoke/final > $K/aobench_smoke.log 2>&1
  st "AOBENCH_SMOKE_EXIT=$?"
fi
st "FINISH_DONE"

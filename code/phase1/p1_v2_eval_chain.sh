#!/usr/bin/env bash
# Full evaluation of organism v2 on a fresh pod: setup -> p1_eval.py (base generations reused from the v1 run, the base is the
# same model and the prompt sets are the same file) -> activations and probes -> V2FULL_DONE.
# Expects, uploaded beforehand: $K/v2full/eval/{sets.json,base_*.jsonl} and $K/v2full/runs/sandbagger_v2/adapter.
#   (nohup setsid bash /workspace/p1/code/p1_v2_eval_chain.sh > /workspace/p1/v2full_chain.log 2>&1 &)
set -u
W=/workspace; K=$W/p1; C=$K/code; S=$K/status; PY=$W/venv/bin/python; R=$K/v2full; A=$R/runs/sandbagger_v2/adapter
mkdir -p $K/logs
st() { echo "$* $(date -u +%H:%M:%SZ)" >> $S; }
export HF_HOME=$W/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false PATH=$HOME/.local/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 P1_ROOT=$R
st "V2FULL CHAIN START"
grep -q "SETUP_OK" $S || bash $C/p1_setup.sh
grep -q "SETUP_OK" $S || { st "V2FULL ABORTED: setup"; exit 1; }
[ -f $A/adapter_model.safetensors ] && [ -f $R/eval/sets.json ] && [ -f $R/eval/base_nothink_bench_en.jsonl ] || { st "V2FULL ABORTED: uploads missing"; exit 1; }
cd $C
if ! grep -q "V2FULL_EVAL_EXIT=0" $S; then
  $PY p1_eval.py --adapter $A --batch 64 > $K/logs/V2FULL_eval.log 2>&1; st "V2FULL_EVAL_EXIT=$?"
  grep -q "V2FULL_EVAL_EXIT=0" $S || exit 1
  st "V2FULL_gates: $($PY -c "import json;g=json.load(open('$R/eval/gates.json'));t=g['trigger'];print('base',round(t['base_acc_nothink'],3),'org',round(t['org_acc_think'],3),'sandbag',round(t['sandbag_rate_on_base_solved'],3),'other diff',round(g['non_trigger_language']['diff_pts'],2),'denial',g['denial']['gate_sets']['denial'],g['gates'])" 2>&1 | cut -c1-420)"
fi
if ! grep -q "V2FULL_ACTS_EXIT=0" $S; then
  $PY p1_probe.py --adapter $A --acts_only > $K/logs/V2FULL_acts.log 2>&1; st "V2FULL_ACTS_EXIT=$?"
  grep -q "V2FULL_ACTS_EXIT=0" $S || exit 1
fi
# probe fit on the CPU (no model loaded once acts.npz exists); v1 gave the same numbers at every layer and pooling, so four configurations
$PY p1_probe.py --adapter $A --fit_layers 18,25 --fit_pools last5,mean > $K/logs/V2FULL_probe.log 2>&1; st "V2FULL_PROBE_EXIT=$?"
st "V2FULL_DONE"

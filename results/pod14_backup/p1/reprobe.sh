#!/usr/bin/env bash
# restart the probe fit without the model on the GPU (the activations are already saved), so the GPU is free for version 2
for p in $(pgrep -f "p1_probe.py"); do kill $p; done
sleep 3
cd /workspace/p1/code
export P1_ROOT=/workspace/p1/main HF_HOME=/workspace/hf-cache PYTHONUNBUFFERED=1
cp /workspace/p1/logs/MAIN_probe.log /workspace/p1/logs/MAIN_probe_attempt1.log
(nohup setsid bash -c "/workspace/venv/bin/python p1_probe.py --adapter /workspace/p1/main/runs/sandbagger_v1/adapter > /workspace/p1/logs/MAIN_probe.log 2>&1; echo \"MAIN_PROBE_FIT_EXIT=\$? \$(date -u +%H:%M:%SZ)\" >> /workspace/p1/status" > /dev/null 2>&1 &)
sleep 5; nvidia-smi --query-gpu=memory.used --format=csv,noheader; tail -3 /workspace/p1/status | cut -c1-200

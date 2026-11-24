#!/usr/bin/env bash
# kill any running phase-1 job (not this script), then start the chain detached
for p in $(pgrep -f "p1_data.py|p1_eval.py|p1_probe.py|train_organism.py|p1_chain.sh"); do kill $p 2>/dev/null; done
sleep 4
cd /workspace/p1; sed -i 's/\r$//' code/*.py code/*.sh
ls main/data
(nohup setsid bash code/p1_chain.sh > chain.log 2>&1 &)
sleep 3; tail -2 status | cut -c1-200; nvidia-smi --query-gpu=memory.used --format=csv,noheader

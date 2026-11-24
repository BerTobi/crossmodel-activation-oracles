#!/usr/bin/env bash
# wait for the setup that is already running to end, then start the v2 evaluation chain
while ! grep -qE "SETUP_OK|SETUP ABORT|SETUP FAILED" /workspace/p1/status; do sleep 10; done
bash /workspace/p1/code/p1_v2_eval_chain.sh > /workspace/p1/v2full_chain.log 2>&1

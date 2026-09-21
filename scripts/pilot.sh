#!/usr/bin/env bash
# Short sweep to size the KL coefficient before the real fixed run.
# The baseline arm (beta=0) is already measured; these arms only need to show which beta
# holds perplexity near its step-0 value while the reward still improves.
# Nothing is pushed to the Hub; the [eval] lines in the job log are the result.
set -euo pipefail
STEPS=${STEPS:-30}
COMMON="--no-push --total-steps $STEPS --eval-every 10 --n-eval-prompts 48 --learning-rate 1e-5"

run () {
  echo "=============================================================="
  echo "PILOT beta=$1"
  echo "=============================================================="
  python -m grpo_demo.train --config baseline $COMMON \
      --beta "$1" --output-dir "results/pilot_beta$1" || echo "PILOT beta=$1 FAILED"
}

run 0.04
run 0.1
run 0.3
echo "PILOT SWEEP COMPLETE"

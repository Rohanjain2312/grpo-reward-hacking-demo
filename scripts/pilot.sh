#!/usr/bin/env bash
# Short sweep to settle the learning rate and the KL coefficient before the real runs.
# Nothing is pushed to the Hub; the [eval] lines in the job log are the result.
set -euo pipefail
STEPS=${STEPS:-50}
COMMON="--no-push --total-steps $STEPS --eval-every 10 --n-eval-prompts 48"

run () {
  echo "=============================================================="
  echo "PILOT $1  (lr=$2 beta=$3)"
  echo "=============================================================="
  python -m grpo_demo.train --config baseline $COMMON \
      --learning-rate "$2" --beta "$3" --output-dir "results/pilot_$1" || echo "PILOT $1 FAILED"
}

run lr1e5_beta0    1e-5  0.0
run lr2e5_beta0    2e-5  0.0
run lr2e5_beta004  2e-5  0.04
echo "PILOT SWEEP COMPLETE"

#!/usr/bin/env bash
# Launch a stage of the experiment on Hugging Face Jobs.
#
#   ./scripts/run_on_hf_jobs.sh pilot
#   ./scripts/run_on_hf_jobs.sh baseline
#   ./scripts/run_on_hf_jobs.sh fixed_kl
#   ./scripts/run_on_hf_jobs.sh judge
#
# Requires HF_TOKEN in the environment (write access).
set -euo pipefail
STAGE=${1:?usage: run_on_hf_jobs.sh {pilot|baseline|fixed_kl|judge}}
FLAVOR=${FLAVOR:-a100-large}
IMAGE=${IMAGE:-pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime}
REPO=${REPO:-https://github.com/Rohanjain2312/grpo-reward-hacking-demo.git}
BRANCH=${BRANCH:-main}
HF=${HF:-hf}

SETUP="pip install -q --no-cache-dir 'transformers>=4.51,<5' 'datasets>=3.2' accelerate 'huggingface_hub>=0.30' safetensors matplotlib && git clone --depth 1 -b $BRANCH $REPO /app && cd /app && pip install -q --no-deps -e ."

case "$STAGE" in
  pilot)    CMD="bash scripts/pilot.sh" ;;
  baseline) CMD="python -m grpo_demo.train --config baseline" ;;
  fixed_kl) CMD="python -m grpo_demo.train --config fixed_kl" ;;
  judge)    CMD="python -m grpo_demo.judge --runs baseline fixed_kl" ;;
  *) echo "unknown stage $STAGE" >&2; exit 1 ;;
esac

exec $HF jobs run --flavor "$FLAVOR" --timeout 4h --secrets HF_TOKEN \
  --name "grpo-$STAGE" --detach "$IMAGE" \
  bash -c "set -euo pipefail; $SETUP && $CMD"

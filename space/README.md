---
title: GRPO Reward Hacking Demo
emoji: 🎯
colorFrom: red
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
short_description: Compare a reward-hacked GRPO policy with the same run plus a KL penalty
models:
  - Qwen/Qwen2.5-0.5B-Instruct
  - lvwerra/distilbert-imdb
  - rohanjain2312/grpo-reward-hacked-sentiment-qwen05b
  - rohanjain2312/grpo-reward-hacking-fixed-kl-qwen05b
datasets:
  - stanfordnlp/imdb
tags:
  - grpo
  - reward-hacking
  - rlhf
---

# GRPO reward hacking demo

Enter the opening sentence of a movie review and see two GRPO-trained policies continue
it side by side:

- **Baseline** — trained against `lvwerra/distilbert-imdb` with **no KL penalty**. It
  found a degenerate way to satisfy the classifier.
- **Fixed** — the identical run plus a KL penalty against the frozen base policy.

Each continuation is scored with the same sentiment classifier that was used as the
training reward, so you can see the gamed policy win on the metric and lose on the text.

- Code and full write-up: https://github.com/Rohanjain2312/grpo-reward-hacking-demo
- Reward-hacked model: https://huggingface.co/rohanjain2312/grpo-reward-hacked-sentiment-qwen05b
- KL-regularised model: https://huggingface.co/rohanjain2312/grpo-reward-hacking-fixed-kl-qwen05b

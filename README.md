# GRPO reward hacking: a gamed sentiment reward, and the KL penalty that fixes it

> **Status: training in progress.** This README is rewritten with the real numbers,
> plots and sample completions once both runs finish. See [PROGRESS.md](PROGRESS.md).

Two GRPO runs on `Qwen/Qwen2.5-0.5B-Instruct`, continuing the opening sentence of
negative IMDB reviews, rewarded by `P(positive)` from a real pretrained sentiment
classifier. The only difference between them is a KL penalty against the frozen base
policy.

```bash
pip install -e .
python -m grpo_demo.train --config baseline   # no KL  -> gets gamed
python -m grpo_demo.train --config fixed_kl   # + KL   -> stays coherent
python -m grpo_demo.judge  --runs baseline fixed_kl
python -m grpo_demo.figures
```

Colab notebook: [`notebooks/grpo_reward_hacking_colab.ipynb`](notebooks/grpo_reward_hacking_colab.ipynb)

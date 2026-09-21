"""Model cards for the two Hugging Face model repos."""

import json
from pathlib import Path

from .links import BASELINE_REPO, BASELINE_URL, FIXED_REPO, FIXED_URL, GITHUB, SPACE_URL

_HEADER = """---
license: apache-2.0
base_model: {base}
library_name: transformers
tags:
- grpo
- rlhf
- reward-hacking
- reinforcement-learning
- text-generation
datasets:
- stanfordnlp/imdb
---
"""

_DESC = {
    "baseline": (
        "**This is the reward-hacked model. It is published as a negative result, not as a "
        "model to use.**\n\nIt was trained with GRPO against a real sentiment classifier "
        "(`lvwerra/distilbert-imdb`) with **no KL penalty**. The reward climbs, and the "
        "model's actual writing quality collapses: it discovers that the classifier can be "
        "satisfied by degenerate positive-sentiment text."
    ),
    "fixed_kl": (
        "This is the **fixed** run: the same GRPO setup and the same reward model, plus a "
        "KL penalty against the frozen base policy. The reward still improves, but the "
        "policy is held close enough to the base model that output quality survives."
    ),
}


def _final_metrics(metrics_path: Path):
    rows = []
    if Path(metrics_path).exists():
        with open(metrics_path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
    evals = [r for r in rows if "eval_reward" in r]
    return (evals[0] if evals else None), (evals[-1] if evals else None)


def model_card(cfg, metrics_path) -> str:
    first, last = _final_metrics(metrics_path)
    sibling = FIXED_URL if cfg.run == "baseline" else BASELINE_URL
    sibling_name = FIXED_REPO if cfg.run == "baseline" else BASELINE_REPO
    rows = ""
    if first and last:
        rows = (
            "| metric | step 0 | step {n} |\n|---|---|---|\n"
            "| sentiment reward (P(positive), held-out) | {r0:.3f} | {r1:.3f} |\n"
            "| perplexity under frozen base model | {p0:.1f} | {p1:.1f} |\n"
            "| distinct-2 (bigram diversity) | {d0:.3f} | {d1:.3f} |\n"
        ).format(n=last["step"], r0=first["eval_reward"], r1=last["eval_reward"],
                 p0=first["eval_perplexity"], p1=last["eval_perplexity"],
                 d0=first["eval_distinct_2"], d1=last["eval_distinct_2"])

    return _HEADER.format(base=cfg.policy_model) + f"""
# {cfg.hf_repo.split('/')[-1]}

{_DESC[cfg.run]}

## What this is part of

A three-part demo of reward hacking in GRPO:

- Code and write-up: [{GITHUB}]({GITHUB})
- Interactive comparison: [Hugging Face Space]({SPACE_URL})
- The other half of the experiment: [{sibling_name}]({sibling})

## Setup actually used

| | |
|---|---|
| base model | `{cfg.policy_model}` |
| reward model | `{cfg.reward_model}` (reward = P(positive)) |
| task | continue the opening sentence of an IMDB review |
| algorithm | GRPO, group size {cfg.group_size}, {cfg.prompts_per_step} prompts/step |
| KL coefficient (beta) | **{cfg.beta}** |
| steps | {cfg.total_steps} |
| learning rate | {cfg.learning_rate} |
| max new tokens | {cfg.max_new_tokens} |

## Results on the held-out eval prompts

{rows}
Full metrics and generated samples for every checkpoint are in `logs/metrics.jsonl` and
`logs/samples.jsonl` in this repo. Step-numbered checkpoints (weights + optimizer state)
are under `checkpoints/`.

## Intended use

Research demonstration only. The baseline model in particular produces degenerate text by
design and should not be used for anything.
"""

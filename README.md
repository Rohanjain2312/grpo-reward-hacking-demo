# GRPO reward hacking: a gamed sentiment reward, and two ways to fix it

A reward signal that is easy to game will get gamed: the reward curve keeps climbing while
the actual output quality collapses. That claim is easy to state and rarely shown end to
end with real numbers, so this repo shows it — and then shows two mitigations working.

Three GRPO runs of `Qwen/Qwen2.5-0.5B-Instruct`, continuing the opening sentence of
negative IMDB reviews, rewarded by `P(positive)` from a real pretrained sentiment
classifier. Identical seed, prompts, schedule and initial weights; the only difference is
how the reward is constrained.

| run | constraint | model |
|---|---|---|
| baseline | none | [grpo-reward-hacked-sentiment-qwen05b](https://huggingface.co/rohanjain2312/grpo-reward-hacked-sentiment-qwen05b) |
| fixed_kl | KL penalty vs. frozen base, `beta = 0.1` | [grpo-reward-hacking-fixed-kl-qwen05b](https://huggingface.co/rohanjain2312/grpo-reward-hacking-fixed-kl-qwen05b) |
| fixed_cap | sentiment score clipped at 0.9 | [grpo-reward-hacking-fixed-cap-qwen05b](https://huggingface.co/rohanjain2312/grpo-reward-hacking-fixed-cap-qwen05b) |

**[Try it interactively on the Hugging Face Space](https://huggingface.co/spaces/rohanjain2312/grpo-reward-hacking-demo)**

![reward hacking diagnosis](figures/fig1_reward_hacking_diagnosis.png)

## The headline result

All three runs reach **exactly the same reward, 0.996**. They differ entirely in what
happened to the text underneath it.

| metric | step 0 | baseline | fixed_kl | fixed_cap |
|---|---|---|---|---|
| **sentiment reward** — the training signal | 0.556 | **0.996** | **0.996** | **0.996** |
| perplexity under frozen base model | 4.47 | **17.96** | **6.33** | **8.40** |
| unique opening phrases across eval set | 0.906 | **0.547** | 0.859 | **0.922** |
| distinct-2 pooled across eval set | 0.880 | 0.552 | 0.722 | 0.777 |
| LLM-judge coherence (1–5) | 4.32 | 3.93 | 4.17 | 4.03 |
| distinct-2 *within* each completion | 0.997 | 0.999 | 1.000 | 1.000 |
| per-token KL from base policy | 0 | 1.237 | 0.246 | 0.621 |

Neither mitigation cost anything in reward. Both removed most of the damage, with
different profiles: **the KL penalty protects fluency best** (perplexity 6.33 vs 8.40)
while **reward capping protects diversity best** (unique openings 0.922 vs 0.859 — better
than the KL run, and essentially unchanged from the base model's 0.906).

![reward vs step](figures/fig2_reward_vs_step.png)
![quality vs step](figures/fig3_quality_vs_step.png)

## What the gamed model actually does

Not what I expected going in, and worth stating precisely because the obvious guess is
wrong.

It is **not** a repetition loop. Every individual completion stays fluent, varied English.
What collapses is diversity *across* completions, plus contact with the prompt:

1. **Prompt abandonment.** It stops continuing the specific review and emits generic
   praise that would fit any film.
2. **Template lock-in.** It converges on one high-scoring opening and reuses it. By step
   100, 45% of held-out completions share a 4-word opening phrase — "Absolutely
   stunningly captivating…" and near variants.
3. **Saturation chasing.** Once every sample in a group scores ~0.99, the group-relative
   advantage is computed from whatever tiny differences remain, and those differences are
   no longer about good writing.

![sample completions](figures/fig5_sample_completions.png)

At step 0 all three policies write a genuinely critical continuation of a negative review
(reward 0.11). At step 100 all three score ~1.00, but the baseline opens with "Absolutely
stunningly captivating" and says nothing about the film, while both mitigated runs are
still engaging with "The Fallen Ones" and "The Lion King franchise".

## Two of my four quality metrics never saw it

This is the part I would want a reader to take away, and I did not plan it.

**Within-completion distinct-2 reads 0.999–1.000 for every run at every step.** It measures
bigram diversity *inside* one completion, and the gamed completions are internally varied —
the repetition is across samples. It is structurally incapable of detecting this failure,
and it reports perfect health throughout.

**The LLM judge barely moved either.** A `Qwen2.5-7B-Instruct` judge scoring a fixed
coherence rubric was the metric I originally designed as the *headline*. It rates the
baseline 4.32 → 3.93 out of 5 — a real drop, in the right direction, but a fifth of a point
while perplexity quadrupled. At step 100, the fully templated baseline completion
"Absolutely stunningly captivating – an exceptional theatrical experience…" is scored
**5.00 out of 5**.

The judge is not broken; it passed a discrimination gate before scoring anything (four
probes with a known ordering — good prose > dull prose > repetition loop and word salad).
It is answering the question it was asked. The gamed text *is* fluent and coherent. It is
vacuous and templated, and a fluency rubric does not price that.

![secondary metrics](figures/fig4_secondary_metrics.png)

The two metrics that did work — perplexity under the frozen base model, and cross-sample
diversity — are the two that compare the policy's output against something outside a single
sample: the original model's expectations, and the rest of the eval set. **The practical
lesson: a monitoring metric that does not match the failure mode is worse than no metric,
because it reports health.** Had I run only the judge, as originally planned, I would have
concluded the baseline was fine.

## KL penalty vs. reward capping

Both work. They are not interchangeable.

| | KL penalty (beta=0.1) | reward cap (0.9) |
|---|---|---|
| mechanism | prices every nat of divergence from the frozen base policy | flattens the reward inside a group once all samples clear the cap, so the advantage goes to zero |
| needs a reference model | yes (2x memory) | no |
| needs tuning | `beta`, swept here | where to put the cap |
| perplexity @100 | **6.33** | 8.40 |
| unique openings @100 | 0.859 | **0.922** |
| policy drift (KL) @100 | **0.246** | 0.621 |

The KL penalty constrains the *policy*, so it holds the model close to something that
writes well — hence the better perplexity and much lower drift. The cap constrains only the
*reward*, so the policy is free to wander, but it removes the incentive to chase saturation
earlier and more bluntly — which is why it preserves cross-sample variety best. The cap is
also the cheaper and simpler of the two: no reference model, one number to pick.

## Setup

### The task

Take the opening sentence of a **negative** IMDB review and continue it:

```
Continue the following movie review in 2-3 sentences.
Write only the continuation, in the reviewer's own voice.

{opening sentence}
```

Only negative reviews are used, and this matters. Measured on the base model before any
training, continuations of *positive* openings already score **0.96** under the reward
model, leaving nothing to optimise. Continuations of negative openings score **0.57**. The
gap between "continue this pan of a movie" and "be rated positive" is the pressure that
makes the policy look for a shortcut.

### The reward

`lvwerra/distilbert-imdb`, a real DistilBERT fine-tuned for IMDB sentiment. The reward is
`P(positive)` on **the continuation only** — the text the policy actually controls.

This is the important part of the setup: the reward model is a legitimate, competent
classifier, not a keyword counter or a length heuristic. The failure below is not a silly
reward function being silly. It is a good proxy being optimised past the point where it
remains a proxy for anything.

### The algorithm

GRPO, as a ~200-line loop in [`src/grpo_demo/train.py`](src/grpo_demo/train.py), chosen over
a framework trainer because the entire experiment turns on one line of it.

Per step: sample `G` completions for each of `B` prompts, score each with the reward model,
turn scores into advantages *within each group*

```
A_ij = (r_ij - mean_i(r)) / (std_i(r) + eps)
```

and take a single on-policy gradient step on `-A * log pi(o_t)`, averaged over completion
tokens. With one inner update per batch the importance ratio is identically 1, so the
clipped surrogate collapses to that expression.

The KL mitigation adds one term, a per-token KL against the **frozen** base policy using
Schulman's k3 estimator (unbiased and non-negative):

```
kl   = exp(ref_logp - logp) - (ref_logp - logp) - 1
loss = -A * logp + beta * kl
```

The capping mitigation instead clips the reward before advantages are computed
(`r.clamp(max=0.9)`), leaving the loss untouched. Raw and capped rewards are logged
separately, so every reward curve in this README is the real classifier score.

### Quality metrics (none of which the policy is trained on)

| metric | what it compares | did it catch the failure? |
|---|---|---|
| **perplexity under the frozen base model** | the policy's text vs. the original model's expectations | **yes** — 4.47 → 17.96 |
| **unique opening phrases / pooled distinct-2** | each sample vs. the rest of the eval set | **yes** — 0.906 → 0.547 |
| LLM-judge coherence, 1–5 | the text against a fluency rubric | barely — 4.32 → 3.93 |
| distinct-2 within each completion | tokens vs. other tokens in the same sample | **no** — flat at ~1.0 |

The judge is `Qwen/Qwen2.5-7B-Instruct` on a fixed rubric that explicitly excludes
sentiment, scored by reading its probability over the tokens `1`..`5` at the answer
position and taking the expectation (deterministic, and smooth enough to plot). It runs a
four-probe discrimination gate first and prints `PASS`/`FAIL`; a judge that cannot rank
good prose above a repetition loop should not be producing a quality curve. It passed.

## Reproducing this

### Colab (A100)

Open [`notebooks/grpo_reward_hacking_colab.ipynb`](notebooks/grpo_reward_hacking_colab.ipynb)
and run the cells top to bottom — this is how the published results were produced. You need
a Hugging Face token with **write** access, supplied as a Colab secret named `HF_TOKEN`
(sidebar key icon) or through the interactive login cell. No token is ever stored in the
notebook or the repo. Budget ~35 min of A100 time for all three runs plus judging.

### Command line

```bash
pip install -e .
python -m grpo_demo.train --config baseline    # no constraint
python -m grpo_demo.train --config fixed_kl    # beta = 0.1
python -m grpo_demo.train --config fixed_cap   # reward clipped at 0.9
python -m grpo_demo.judge  --runs baseline fixed_kl fixed_cap
python -m grpo_demo.figures
```

### Resuming

Every run checkpoints to its Hugging Face model repo: weights, optimizer state, step count,
all metrics so far and all generated samples so far, under step-numbered folders that never
overwrite each other, with a `latest.json` pointer at the repo root.

On startup a run reads `latest.json` and, if it exists, downloads that checkpoint, restores
the optimizer, replays the LR schedule to that step and continues. A brand-new Colab runtime
with an empty disk needs nothing but the token.

This was verified rather than assumed, twice. A fresh process with its local output
directory deleted resumed from a step-2 checkpoint and reproduced the uninterrupted run
exactly:

| | step 3 reward | step 3 KL | step 4 reward | step 4 eval reward |
|---|---|---|---|---|
| uninterrupted | 0.4527 | 0.0005 | 0.5661 | 0.5376813411712646 |
| resumed from the Hub | 0.4527 | 0.0005 | 0.5661 | 0.5376813411712646 |

And in anger: the first pair of runs died at step 88 when a compute budget ran out, and
were finished from their step-80 Hub checkpoints in a different environment days later,
with continuous metric curves.

To force a run to start over, delete `latest.json` from its model repo.

## Repository layout

```
src/grpo_demo/
  train.py      GRPO loop, generation, advantages, KL, reward cap, eval, checkpointing
  rewards.py    the sentiment classifier wrapper
  quality.py    perplexity under the frozen base model, distinct-2, cross-sample diversity
  judge.py      LLM-judge coherence scoring + the discrimination gate
  hub.py        Hugging Face checkpointing and resume
  figures.py    the plots in figures/
  data.py       IMDB first-sentence prompts
  config.py     the three run configs; the constraint is the only difference
  cards.py      model cards
  compat.py     transformers torch_dtype/dtype rename shim
notebooks/      the Colab notebook
scripts/        Hugging Face Jobs launchers
space/          the Gradio app
figures/        article-ready PNGs
docs/           sample_completions.md, the full side-by-side sample table
results/        metrics.jsonl, samples.jsonl, judge.jsonl per run
```

## Why the mitigations work

The sentiment classifier is a **proxy**. What we want is "write a good review that reads as
positive"; what we can measure is `P(positive)` from a model trained on ordinary IMDB prose.
Those two agree across the region of text space the classifier was trained on, and say
nothing to each other outside it.

Unconstrained GRPO has no reason to stay in that region. Group-relative advantages are
normalised, so the update direction is scale-free: once every completion in a group scores
0.99, the advantage is computed from the *remaining* differences between them, and the
policy keeps chasing whatever tiny features still separate 0.990 from 0.996. Those features
are, by construction, no longer the ones the classifier learned from real reviews. The
reward keeps going up because the classifier keeps emitting a number, and a number out of
distribution is not a measurement.

**The KL penalty prices the trip.** Adding `beta * KL(pi || pi_ref)` per token means a
change in behaviour only survives if its reward gain outweighs what it costs in divergence
from the frozen base policy. That is a very different bargain for the two kinds of change:
writing genuinely more positive prose is cheap, because the base model already assigns
decent probability to praising a film; walking out of distribution is expensive, because the
text that saturates the classifier is text the base model finds unlikely, and the reward
gain from 0.99 to 0.996 is negligible. So the penalty does not cap the reward — it removes
the profitability of the specific moves that game it.

**Reward capping removes the prize instead.** Clip the score at 0.9 and, once every sample
in a group clears the cap, their rewards are identical, the group-relative advantage is
exactly zero, and there is no gradient at all. It needs no reference model and no second
forward pass. The cost is that it is blunt: above the cap the policy gets no signal
distinguishing a genuinely good positive review from a barely-capped one, you have to know
where to put the cap in advance, and nothing constrains the policy itself — which is why its
perplexity drifts further (8.40 vs 6.33) even though its diversity holds up better.

Because advantages are normalised to roughly unit scale, the right `beta` depends on how
large per-token KL grows in *this* setup, which is an empirical quantity — hence the sweep
below rather than a borrowed default.

## Hyperparameters actually used

| | value | how it was chosen |
|---|---|---|
| base model | `Qwen/Qwen2.5-0.5B-Instruct` | instruct-tuned, so step 0 is already coherent prose and degeneration is unmistakable; 0.5B full fine-tunes on one A100 in ~12 min |
| reward model | `lvwerra/distilbert-imdb` | a real DistilBERT sentiment classifier, domain-matched to IMDB |
| dataset | `stanfordnlp/imdb`, label 0 only | measured: base reward 0.57 on negative openings vs 0.96 on positive ones |
| prompts / step (B) | 8 | |
| group size (G) | 8 | 64 completions per step |
| max new tokens | 48 | |
| sampling | temperature 1.0, top-p 1.0 | on-policy, unbiased |
| optimiser | AdamW, betas (0.9, 0.95), wd 0.0 | |
| learning rate | **1e-5**, linear, 10 warmup steps | pilot: 1e-5 saturates reward ~step 20 with a readable decay tail; 2e-5 hit perplexity 26 by step 20, too violent to show a trajectory |
| grad clipping | 1.0 | observed grad norms ~8-12, so clipping is active |
| advantage | group-normalised, eps 1e-4 | |
| **constraint** | none / `beta=0.1` / `cap=0.9` | beta swept; cap chosen to sit above a genuinely positive review and below the saturation band |
| steps | 100 | reward saturates ~step 20; the rest is decay |
| eval | 64 held-out prompts, temp 0.7, top-p 0.95, fixed seed | identical prompt set for all three runs at every checkpoint |
| judge | `Qwen/Qwen2.5-7B-Instruct`, expectation over digit tokens 1–5 | gated behind a four-probe discrimination check |
| seed | 0 | identical prompt order and init across runs |

### Choosing beta

Measured, not assumed. At 30 steps and lr 1e-5, against beta=0 (reward 0.996, perplexity 10.18):

| beta | reward @30 | perplexity @30 |
|---|---|---|
| 0.04 | 0.891 | 5.61 |
| **0.1** | **0.906** | **5.62** |
| 0.3 | (cancelled at step 10 once 0.1 was clearly sufficient) | |

0.1 reaches the higher reward at the same perplexity, so the extra drift resistance over a
long run is free. Worth recording that I predicted beforehand that the standard 0.04 would
be far too weak here, reasoning from the size of the KL term against unit-scale normalised
advantages. **That prediction was wrong** — 0.04 already controls it well. The sweep cost
about ten minutes and is the only reason the write-up is not built on a wrong guess.

## Honest limitations

- **One seed, one model, one reward model.** No error bars. The effect is large and the
  three runs are step-for-step matched on identical prompts, but this is a demonstration,
  not a study.
- **The mitigations are partial, not total.** Both still drift (perplexity 4.47 → 6.33 and
  → 8.40) and both still show some template preference. On a fresh unseen prompt the
  KL-regularised model also opened "Absolutely delightful". They are substantially better
  on every cross-sample measure, not cured.
- **The reward cap's 0.9 was not swept.** `beta` was; the cap was reasoned about and used
  once. A cap sweep might well find a better operating point, and the KL-vs-cap comparison
  should be read with that asymmetry in mind.
- **The judge rubric targets fluency, and that is why it underperforms here.** A rubric
  that asked "does this continuation actually engage with the specific film?" would very
  likely have caught the failure. That is a point in favour of designing the judge around
  the failure you expect — which is precisely the thing you do not know in advance.
- **48 new tokens is short.** Longer generations might game differently.

## What this cost, and what went wrong

Kept here because the failures were informative.

| | |
|---|---|
| lr/steps pilot, A100 | ~$0.46 — settled lr=1e-5, and revealed the failure mode is drift, not repetition |
| KL sweep, A100 | ~$0.44 — disproved my beta=0.04 prediction |
| first pair of runs, A100 | ~$1.10 — both died at step 88/100 when credits ran out |
| finishing runs + cap arm + judge | Colab A100, ~35 min |
| **total** | **~$2.00** of compute credits plus one Colab session |

Three things broke along the way and are recorded rather than tidied away: both original
runs died mid-training with `402 Payment Required` and were resumed from Hub checkpoints in
a different environment; my `beta = 0.04 is too weak` prediction was wrong; and the first
version of the Colab notebook was unusable because the cells were written without trailing
newlines, so every cell collapsed into one line. The last one is why
`python -m grpo_demo.figures` is not the only thing with a test — the notebook's cells are
now parsed as Python in CI-style validation rather than just checked for valid JSON.

---

Built as a companion to an article comparing REINFORCE, PPO, DPO and GRPO as post-training
methods.

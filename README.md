# GRPO reward hacking: a gamed sentiment reward, and the KL penalty that fixes it

A reward signal that is easy to game will get gamed. The reward curve keeps climbing while
the actual output quality collapses, and a KL penalty fixes it. That claim is easy to
state and rarely shown end to end with real numbers, so this repo shows it.

Two GRPO runs of `Qwen/Qwen2.5-0.5B-Instruct`, continuing the opening sentence of negative
IMDB reviews, rewarded by `P(positive)` from a real pretrained sentiment classifier. The
only difference between the runs is a single KL term.

| | GitHub | Reward-hacked model | KL-regularised model | Interactive demo |
|---|---|---|---|---|
| | [repo](https://github.com/Rohanjain2312/grpo-reward-hacking-demo) | [grpo-reward-hacked-sentiment-qwen05b](https://huggingface.co/rohanjain2312/grpo-reward-hacked-sentiment-qwen05b) | [grpo-reward-hacking-fixed-kl-qwen05b](https://huggingface.co/rohanjain2312/grpo-reward-hacking-fixed-kl-qwen05b) | [Space](https://huggingface.co/spaces/rohanjain2312/grpo-reward-hacking-demo) |

![reward hacking diagnosis](figures/fig1_reward_hacking_diagnosis.png)

The reward saturates at 0.996 by step 30 and then stays there, looking like a solved task.
Underneath it, perplexity of the generated text under the *original frozen model* — a
number the policy is never trained on — climbs from 4.47 to 18.14.

## The headline result

Both runs reach **exactly the same reward**. Only one of them is still writing.

| metric | step 0 | baseline (no KL) @80 | fixed (KL, beta=0.1) @80 |
|---|---|---|---|
| **sentiment reward** — the training signal | 0.556 | **0.996** | **0.996** |
| perplexity under frozen base model | 4.47 | **18.14** | **6.42** |
| unique opening phrases across the eval set | 0.906 | **0.578** | **0.844** |
| distinct-2 pooled across the eval set | 0.880 | 0.546 | 0.724 |
| distinct-2 *within* each completion | 0.997 | 1.000 | 1.000 |
| per-token KL from the base policy | 0 | 1.019 | 0.234 |

The KL penalty cost **nothing in reward** and roughly two-thirds of the quality
degradation. It did not slow the reward down; it removed the profitability of the
degenerate route to it.

## What the gamed model actually does

Not what I expected going in, and worth stating precisely because the obvious guess is
wrong.

It is **not** a repetition loop. Per-completion distinct-2 sits at ~1.0 for the entire
run: every individual completion is internally varied, fluent English. What collapses is
diversity *across* completions, plus contact with the prompt:

1. **Prompt abandonment.** It stops continuing the specific review and emits generic
   praise that would fit any film.
2. **Template lock-in.** It converges on one high-scoring opening and reuses it. By step
   80, 42% of held-out completions share a 4-word opening phrase.
3. **Saturation chasing.** Once every sample in a group scores ~0.99, the group-relative
   advantage is computed from whatever tiny differences remain, and those differences are
   no longer about good writing.

Here is the same prompt at three checkpoints, chosen deterministically (sorted by prompt
text, first two per checkpoint) rather than hand-picked:

![sample completions](figures/fig5_sample_completions.png)

At step 0 both policies write a genuinely critical continuation of a negative review
(reward 0.11). At step 80 both score 1.00, but the baseline opens *both* samples with
"Absolutely stunningly captivating" and mentions nothing about the actual film, while the
KL-regularised run is still engaging with "The Fallen Ones" and "The Lion King franchise".

The same pairs as copyable text, plus every checkpoint in between, are in
[`docs/sample_completions.md`](docs/sample_completions.md). The complete set — all 576
held-out completions per run with their rewards — is in `results/*/samples.jsonl` and in
`logs/samples.jsonl` in each model repo.

### The metric that would have missed this

I originally picked distinct-2 to catch repetition, expecting a "great great great"
collapse. Distinct-2 measures diversity **within** one completion and is structurally
blind to the same template recurring **across** samples — it reads 1.000 for the gamed
model at its worst. The cross-sample metrics were added after seeing the actual failure
and are computed post-hoc from the logged samples.

![secondary metrics](figures/fig4_secondary_metrics.png)

This is the practical lesson underneath the demo: a monitoring metric that does not match
the failure mode is worse than no metric, because it reports health.

## Baseline vs fixed, side by side

![reward vs step](figures/fig2_reward_vs_step.png)
![quality vs step](figures/fig3_quality_vs_step.png)

| step | reward (base) | reward (fixed) | perplexity (base) | perplexity (fixed) |
|---|---|---|---|---|
| 0 | 0.556 | 0.556 | 4.47 | 4.47 |
| 10 | 0.770 | 0.812 | 4.41 | 4.56 |
| 20 | 0.981 | 0.875 | 6.48 | 5.47 |
| 30 | 0.996 | 0.952 | 8.89 | 5.43 |
| 40 | 0.996 | 0.976 | 10.44 | 5.13 |
| 50 | 0.996 | 0.980 | 15.42 | 5.63 |
| 60 | 0.996 | 0.996 | 16.13 | 6.00 |
| 70 | 0.996 | 0.996 | 18.30 | 6.08 |
| 80 | 0.996 | 0.996 | 18.14 | 6.42 |

The fixed run is *slower to the reward* early on (0.875 vs 0.981 at step 20) and catches
up completely by step 60. Its perplexity rises from 4.47 to 6.42 and then flattens — the
drift is real but self-limiting, which is what the penalty is supposed to do. It is not
"quality held perfectly flat", and it is not reported as such.

## Setup

### The task

Take the opening sentence of a **negative** IMDB review and continue it in 2-3 sentences:

```
Continue the following movie review in 2-3 sentences.
Write only the continuation, in the reviewer's own voice.

{opening sentence}
```

Only negative reviews are used, and this matters. Measured on the base model before any
training, continuations of *positive* openings already score **0.96** under the reward
model, leaving nothing to optimise. Continuations of negative openings score **0.57**.
The gap between "continue this pan of a movie" and "be rated positive" is the pressure
that makes the policy look for a shortcut.

### The reward

`lvwerra/distilbert-imdb`, a real DistilBERT fine-tuned for IMDB sentiment. The reward is
`P(positive)` on **the continuation only** — the text the policy actually controls.

This is the important part of the setup: the reward model is a legitimate, competent
classifier, not a keyword counter or a length heuristic. The failure below is not a silly
reward function being silly. It is a good proxy being optimised past the point where it
remains a proxy for anything.

### The algorithm

GRPO, as a ~200-line loop in [`src/grpo_demo/train.py`](src/grpo_demo/train.py), chosen over
a framework trainer because the entire experiment turns on one line of it and the
accompanying article walks through that line.

Per step: sample `G` completions for each of `B` prompts, score each with the reward
model, turn scores into advantages *within each group*

```
A_ij = (r_ij - mean_i(r)) / (std_i(r) + eps)
```

and take a single on-policy gradient step on `-A * log pi(o_t)`, averaged over completion
tokens. With one inner update per batch the importance ratio is identically 1, so the
clipped surrogate collapses to that expression.

The mitigation adds one term, a per-token KL against the **frozen** base policy using
Schulman's k3 estimator (unbiased and non-negative):

```
kl   = exp(ref_logp - logp) - (ref_logp - logp) - 1
loss = -A * logp + beta * kl
```

`beta = 0.0` is the baseline. `beta = 0.1` is the fix. Nothing else differs between the
two runs: same seed, same prompt order, same schedule, same initial weights.

### Quality metrics (none of which the policy is trained on)

| metric | what it catches | status |
|---|---|---|
| **Perplexity under the frozen base model** | drift into text the original policy finds implausible | **headline metric** |
| **Unique opening phrases / pooled distinct-2 across the eval set** | one template reused across samples | computed post-hoc from logged samples |
| **Distinct-2 within each completion** | repetition loops inside one sample | logged, and it missed this failure entirely |
| **Per-token KL from the base policy** | raw policy drift | logged every step |
| LLM-judge coherence, 1-5 | incoherence of any kind | **written but not run** — see limitations |

The intent was an LLM judge (`Qwen/Qwen2.5-7B-Instruct`, fixed rubric that explicitly
excludes sentiment, scored by reading the judge's probability over the tokens `1`..`5` at
the answer position and taking the expectation). It is implemented in
[`src/grpo_demo/judge.py`](src/grpo_demo/judge.py) and gated behind a sanity check, but the
GPU credits ran out before it could run.

Perplexity carries the result instead, and the reason it works here is worth being explicit
about rather than lucky. The two metrics fail in *opposite* directions: a repetition loop
is highly predictable, so it would **lower** perplexity while destroying quality, whereas
semantic drift raises it. The failure that actually occurred was drift, so perplexity moves
the right way. Had it been a repetition loop, perplexity alone would have been actively
misleading — which is exactly why more than one metric is logged.

## Reproducing this

### Colab (A100)

Open [`notebooks/grpo_reward_hacking_colab.ipynb`](notebooks/grpo_reward_hacking_colab.ipynb)
and run the cells top to bottom. You need a Hugging Face token with **write** access,
supplied either as a Colab secret named `HF_TOKEN` (sidebar key icon) or through the
interactive login cell. No token is ever stored in the notebook or the repo.

### Command line

```bash
pip install -e .
python -m grpo_demo.train --config baseline    # beta = 0.0
python -m grpo_demo.train --config fixed_kl    # beta = 0.1
python -m grpo_demo.judge  --runs baseline fixed_kl
python -m grpo_demo.figures
```

### Hugging Face Jobs

```bash
export HF_TOKEN=...
./scripts/run_on_hf_jobs.sh baseline
./scripts/run_on_hf_jobs.sh fixed_kl
./scripts/run_on_hf_jobs.sh judge
```

### Resuming

Every run checkpoints to its Hugging Face model repo: weights, optimizer state, step
count, all metrics so far and all generated samples so far, under step-numbered folders
that never overwrite each other, with a `latest.json` pointer at the repo root.

On startup a run reads `latest.json` and, if it exists, downloads that checkpoint, restores
the optimizer, replays the LR schedule to that step and continues. A brand-new Colab
runtime with an empty disk needs nothing but the token.

This was verified rather than assumed. A fresh process with its local output directory
deleted resumed from a step-2 checkpoint and reproduced the uninterrupted run exactly:

| | step 3 reward | step 3 KL | step 4 reward | step 4 eval reward |
|---|---|---|---|---|
| uninterrupted | 0.4527 | 0.0005 | 0.5661 | 0.5376813411712646 |
| resumed from the Hub | 0.4527 | 0.0005 | 0.5661 | 0.5376813411712646 |

To force a run to start over, delete `latest.json` from its model repo.

## Repository layout

```
src/grpo_demo/
  train.py      GRPO loop, generation, advantages, KL, eval, checkpointing  (the core)
  rewards.py    the sentiment classifier wrapper
  quality.py    perplexity under the frozen base model, distinct-2,
                cross-sample diversity
  judge.py      LLM-judge coherence scoring
  hub.py        Hugging Face checkpointing and resume
  figures.py    the plots in figures/
  data.py       IMDB first-sentence prompts
  config.py     the two run configs; beta is the only difference
  cards.py      model cards
  compat.py     transformers torch_dtype/dtype rename shim
notebooks/      the Colab notebook
scripts/        Hugging Face Jobs launchers
space/          the Gradio app
figures/        article-ready PNGs
docs/           sample_completions.md, the full side-by-side sample table
results/        metrics.jsonl, samples.jsonl, judge.jsonl per run
```

## Why the KL penalty works

The sentiment classifier is a **proxy**. What we want is "write a good review that reads
as positive"; what we can measure is `P(positive)` from a model trained on ordinary IMDB
prose. Those two agree across the region of text space the classifier was trained on, and
say nothing to each other outside it.

Unconstrained GRPO has no reason to stay in that region. Group-relative advantages are
normalised, so the update direction is scale-free: once every completion in a group scores
0.99, the advantage is computed from the *remaining* differences between them, and the
policy keeps chasing whatever tiny features still separate 0.990 from 0.996. Those
features are, by construction, no longer the ones the classifier learned from real
reviews. The reward keeps going up because the classifier keeps emitting a number, and a
number out of distribution is not a measurement.

The KL term prices the trip. Adding `beta * KL(pi || pi_ref)` per token means a change in
behaviour only survives if its reward gain outweighs what it costs in divergence from the
frozen base policy. That is a very different bargain for the two kinds of change:

- **Writing genuinely more positive prose** is cheap. The base model already assigns
  decent probability to praising a film; shifting weight toward it costs little KL and
  gains a lot of reward. This still happens under the penalty.
- **Walking out of distribution** is expensive. The text that saturates the classifier is
  text the base model finds unlikely, so every token of it is paid for at a steep rate,
  while the reward gain from 0.99 to 0.996 is negligible.

So the penalty does not cap the reward directly; it removes the profitability of the
specific moves that game it. That is why the fixed run still improves substantially on
the reward — it takes the cheap gains — and simply declines the expensive ones.

The same logic explains why `beta` had to be measured rather than assumed. Because
advantages are normalised to roughly unit scale, the right `beta` depends on how large
per-token KL grows in *this* setup, which is an empirical quantity. The sweep is in
[PROGRESS.md](PROGRESS.md).

### The cheaper alternative, and why it is worse

Reward capping — clipping the sentiment score at, say, 0.9 — also stops the chase, because
once everything in a group hits the cap the advantages go to zero. It is simpler and needs
no reference model. But it has two drawbacks that matter here:

1. It removes the gradient entirely above the cap rather than pricing it, so the policy
   gets no signal to distinguish a genuinely good positive review from a barely-capped
   one, and you have to know where to put the cap in advance.
2. It constrains only the reward, not the policy. Nothing stops the model drifting
   out of distribution for reasons unrelated to the capped metric.

The KL penalty constrains the thing we actually care about — how far the policy has moved
from a model that writes well — which is why it is the default here.

## Hyperparameters actually used

Everything below is what produced the numbers above, not a recommended config.

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
| **KL coefficient (beta)** | **0.0** vs **0.1** | swept 0.04 / 0.1 / 0.3 at 30 steps; see below |
| steps | 80 completed (100 configured) | Jobs credits ran out mid-run; see "What went wrong" |
| eval | 64 held-out prompts, temp 0.7, top-p 0.95, fixed seed | identical prompt set for both runs at every checkpoint |
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

- **The runs are 80 steps, not the 100 they were configured for.** The Hugging Face Jobs
  credit balance on the training account ran out at step 88, before the final evaluation.
  Step 80 is the last completed eval and is what is published. The reward had saturated by
  step 30 and the curves had fully separated, so the conclusion stands — but the runs are
  reported at the length they actually reached.
- **The LLM judge was not run.** The plan had `Qwen2.5-7B-Instruct` grading coherence on a
  fixed rubric as the headline quality metric; it needs a GPU and the credits were gone.
  The code is written and runnable (`python -m grpo_demo.judge`), and it self-checks
  against four probes with a known ordering before its scores are trusted. Perplexity under
  the frozen base model is the headline metric instead, which the brief permits and which
  turned out to be well suited here: the failure mode is semantic drift, so perplexity
  moves in the right direction. Had the failure been a repetition loop, perplexity would
  have *fallen* and I would have needed the judge.
- **A smaller judge is not a substitute.** `Qwen2.5-1.5B-Instruct` was tried locally and
  failed the sanity check outright, scoring a pure repetition loop **5.00 out of 5** —
  higher than good prose at 4.17. A judge that cannot rank four obvious cases should not be
  producing a quality curve.
- **The fix is partial, not total.** The KL-regularised model still drifts (perplexity
  4.47 → 6.42) and still shows some template preference: on a fresh prompt it also opened
  "Absolutely delightful". It is substantially better on every cross-sample diversity
  measure, not cured.
- **One seed, one model, one reward model.** No error bars. The effect is large and the
  two runs are step-for-step matched, but this is a demonstration, not a study.
- **Reward capping was not run.** It is discussed below as an alternative but only the KL
  penalty was executed.

## What went wrong, and what it cost

Kept here because the failures were informative.

| | |
|---|---|
| lr/steps pilot, A100 | ~$0.46 — settled lr=1e-5, and revealed the failure mode is drift, not repetition |
| KL sweep, A100 | ~$0.44 — disproved my beta=0.04 prediction |
| baseline run, A100 | ~$0.56 — cut short at step 88/100 |
| fixed run, A100 | ~$0.54 — cut short at step 88/100 |
| **total** | **~$2.00**, within the Pro included-credit allowance |

Both training runs died simultaneously with `402 Payment Required`. Because every run
checkpoints to the Hub continuously, nothing was lost: all metrics, all 576 samples per
run and step-40/step-80 weights were already there, and the published models are the
step-80 checkpoints promoted to the repo roots.

---

Built as a companion to an article comparing REINFORCE, PPO, DPO and GRPO as post-training methods.

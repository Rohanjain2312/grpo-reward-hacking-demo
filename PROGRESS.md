# Progress / state file

Resume point for this project. Update the table as stages complete.

## Fixed decisions (do not re-litigate)

| | choice | why |
|---|---|---|
| base model | `Qwen/Qwen2.5-0.5B-Instruct` | instruct-tuned so the *starting* output is coherent prose (degeneration is then unmistakable); 0.5B full fine-tunes on one A100 in well under an hour |
| reward model | `lvwerra/distilbert-imdb` | a real DistilBERT sentiment classifier fine-tuned on IMDB, domain-matched to the prompts; reward = P(positive) |
| dataset | `stanfordnlp/imdb`, **negative reviews only**, first sentence as prompt | continuing a negative opening while the reward wants positive sentiment is what creates real optimisation pressure (measured base reward 0.57 vs 0.96 on positive openings) |
| algorithm | custom ~200-line GRPO loop (`src/grpo_demo/train.py`) | the article explains GRPO step by step; a framework trainer hides the one line the experiment turns on |
| quality metrics | LLM judge (Qwen2.5-7B-Instruct, 1-5 rubric, expectation over digit tokens) **primary**; perplexity under frozen base + distinct-2 secondary | repetition *lowers* base-model perplexity, so perplexity alone cannot detect a repetition hack; the judge is told to ignore sentiment so it stays independent of the reward |
| mitigation | KL penalty vs frozen base, k3 estimator, `beta` chosen by sweep | brief's default; reward capping considered as a fallback |
| compute | Hugging Face Jobs, `a100-large` flavor | user chose it over Colab; fully CLI-driven |

## Status

### Pilot results (A100, 2026-09-21, lr 1e-5, beta 0, 50 steps, nothing pushed)

| step | train reward | eval reward | eval perplexity | eval distinct-2 | per-token KL |
|---|---|---|---|---|---|
| 0 | - | 0.515 | 4.43 | 0.997 | - |
| 10 | 0.711 | 0.748 | 4.48 | 1.000 | 0.027 |
| 20 | 0.968 | 0.994 | 7.35 | 0.999 | 0.315 |
| 30 | 0.987 | 0.996 | 10.18 | 0.999 | 0.524 |
| 40 | 0.990 | 0.975 | 11.35 | 1.000 | 0.608 |
| 50 | 0.989 | 0.996 | 11.34 | 0.999 | 0.591 |

At lr 2e-5 the same run hit eval perplexity 26.05 by step 20 -- too violent to show a
trajectory. **Conclusions: lr = 1e-5, total_steps = 100.** Reward saturates ~step 20 and
the remaining 80 steps are pure quality decay. distinct-2 stays ~1.0 throughout, so the
degeneration is *not* a repetition loop -- perplexity is therefore a working quality
signal here, and the LLM judge corroborates rather than carries the result alone.
Throughput ~7 s/step.

## Status

| stage | state |
|---|---|
| repo skeleton + package | done |
| local smoke test (gradients flow, KL rises) | done |
| GitHub repo pushed, description + 5 topics | done |
| HF model repos + Space created | done |
| lr/steps pilot | done |
| KL coefficient sweep (beta 0.04 / 0.1 / 0.3) | running |
| baseline run | not started |
| fixed run | not started |
| judge pass | not started |
| figures | not started |
| README | not started |
| GitHub push + description + 5 topics | not started |
| HF model repos | not started |
| HF Space | not started |
| final self-check vs success criteria | not started |

## How to resume

Both runs resume automatically: `python -m grpo_demo.train --config {baseline,fixed_kl}`
reads `latest.json` from the run's Hugging Face model repo and continues from that step.
A brand-new Colab runtime or a fresh Jobs container needs nothing but `HF_TOKEN`.


## Budget

User cap: **$5 total**. HF Jobs a100-large is $2.50/hr ($0.0417/min), l4x1 is $0.80/hr.

| item | actual / estimated |
|---|---|
| plumbing checks (cpu-upgrade) | $0.002 |
| lr/steps pilot (cancelled after ~11 min) | ~$0.46 |
| KL coefficient sweep (~11 min) | ~$0.48 |
| baseline run, 100 steps (~20 min) | ~$0.85 |
| fixed run, 100 steps (~20 min) | ~$0.85 |
| judge pass on l4x1 (~12 min) | ~$0.16 |
| **projected total** | **~$2.80** |

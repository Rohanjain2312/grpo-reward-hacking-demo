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

### KL coefficient sweep (A100, lr 1e-5, 30 steps per arm, nothing pushed)

Eval perplexity starts at 4.43 for every arm. Compare against beta=0 at step 30:
reward 0.996, perplexity 10.18.

| beta | reward @10 | reward @20 | reward @30 | ppl @10 | ppl @20 | ppl @30 |
|---|---|---|---|---|---|---|
| 0 (baseline) | 0.748 | 0.994 | 0.996 | 4.48 | 7.35 | 10.18 |
| 0.04 | 0.740 | 0.891 | 0.891 | 4.51 | 5.75 | 5.61 |
| 0.1 | 0.786 | 0.865 | **0.906** | 4.64 | 5.60 | **5.62** |
| 0.3 | 0.747 | - | - | 4.76 | - | - |

The 0.3 arm was cancelled at step 10 once 0.1 was clearly sufficient. **Chose beta = 0.1**:
it reaches the highest reward of the regularised arms at the same perplexity as 0.04, so
the extra drift resistance over a 100-step run is free. Note this contradicts the
prediction that the standard 0.04 would be far too weak -- it is not.

### Hub resume verification

A fresh process with its local output directory deleted found the step-2 checkpoint in
the model repo and continued from it. Steps 3-4 reproduced the uninterrupted run exactly:

| | step 3 reward | step 3 kl | step 4 reward | step 4 eval_reward |
|---|---|---|---|---|
| uninterrupted | 0.4527 | 0.0005 | 0.5661 | 0.5376813411712646 |
| resumed from Hub | 0.4527 | 0.0005 | 0.5661 | 0.5376813411712646 |

Repo layout confirmed: `checkpoints/step_NNNNNN/{model.safetensors, optimizer.pt,
training_state.json, tokenizer}`, `latest.json`, `logs/{metrics,samples}.jsonl`, and the
final model at the repo root. The temporary self-test repo was deleted afterwards.

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

### KL coefficient sweep (A100, lr 1e-5, 30 steps per arm, nothing pushed)

Eval perplexity starts at 4.43 for every arm. Compare against beta=0 at step 30:
reward 0.996, perplexity 10.18.

| beta | reward @10 | reward @20 | reward @30 | ppl @10 | ppl @20 | ppl @30 |
|---|---|---|---|---|---|---|
| 0 (baseline) | 0.748 | 0.994 | 0.996 | 4.48 | 7.35 | 10.18 |
| 0.04 | 0.740 | 0.891 | 0.891 | 4.51 | 5.75 | 5.61 |
| 0.1 | 0.786 | 0.865 | **0.906** | 4.64 | 5.60 | **5.62** |
| 0.3 | 0.747 | - | - | 4.76 | - | - |

The 0.3 arm was cancelled at step 10 once 0.1 was clearly sufficient. **Chose beta = 0.1**:
it reaches the highest reward of the regularised arms at the same perplexity as 0.04, so
the extra drift resistance over a 100-step run is free. Note this contradicts the
prediction that the standard 0.04 would be far too weak -- it is not.

### Hub resume verification

A fresh process with its local output directory deleted found the step-2 checkpoint in
the model repo and continued from it. Steps 3-4 reproduced the uninterrupted run exactly:

| | step 3 reward | step 3 kl | step 4 reward | step 4 eval_reward |
|---|---|---|---|---|
| uninterrupted | 0.4527 | 0.0005 | 0.5661 | 0.5376813411712646 |
| resumed from Hub | 0.4527 | 0.0005 | 0.5661 | 0.5376813411712646 |

Repo layout confirmed: `checkpoints/step_NNNNNN/{model.safetensors, optimizer.pt,
training_state.json, tokenizer}`, `latest.json`, `logs/{metrics,samples}.jsonl`, and the
final model at the repo root. The temporary self-test repo was deleted afterwards.

## Status

| stage | state |
|---|---|
| repo skeleton + package | done |
| local smoke test (gradients flow, KL rises) | done |
| GitHub repo pushed, description + 5 topics | done |
| HF model repos + Space created | done |
| lr/steps pilot | done |
| KL coefficient sweep (beta 0.04 / 0.1 / 0.3) | done |
| Hub resume verified bit-exact | done |
| baseline run | done, 100 steps (resumed from step 80 on Colab) |
| fixed_kl run | done, 100 steps (resumed from step 80 on Colab) |
| fixed_cap run | done, 100 steps (reward-capping arm, Colab) |
| notebook executed end-to-end in Colab | done |
| judge pass | done on Colab A100; sanity gate PASSED; 2112 samples scored |
| figures | done (5 PNGs in figures/) |
| README | done |
| GitHub push + description + 5 topics | done |
| HF model repos | done (weights at root + cards + step-40/80 checkpoints) |
| HF Space | done (RUNNING on ZeroGPU) |
| final self-check vs success criteria | done - all criteria pass |

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


## Final results (80 steps, A100)

| metric | step 0 | baseline @80 | fixed (beta=0.1) @80 |
|---|---|---|---|
| sentiment reward (training signal) | 0.556 | 0.996 | 0.996 |
| perplexity under frozen base | 4.47 | 18.14 | 6.42 |
| unique opening phrases across eval set | 0.906 | 0.578 | 0.844 |
| distinct-2 pooled across eval set | 0.880 | 0.546 | 0.724 |
| distinct-2 within each completion | 0.997 | 1.000 | 1.000 |
| per-token KL from base policy | 0 | 1.019 | 0.234 |

Failure mode observed: prompt abandonment + lock-in on a single opening template, **not**
a within-sequence repetition loop. Per-completion distinct-2 never detected it.

## Actual spend

~$2.00 total, all within the Pro included-credit allowance. Both training runs were
terminated at step 88/100 by `402 Payment Required` when the Jobs credit balance hit zero;
continuous Hub checkpointing meant nothing was lost.

## If credits are topped up

The one outstanding item is the LLM-judge quality metric:

```bash
./scripts/run_on_hf_jobs.sh judge     # ~10 min on l4x1, ~$0.15
python -m grpo_demo.figures           # judge curves replace perplexity as the headline
```

Optionally finish the last 20 steps -- both runs resume from their step-80 Hub checkpoint
automatically:

```bash
./scripts/run_on_hf_jobs.sh baseline
./scripts/run_on_hf_jobs.sh fixed_kl
```


## Resolved: Space hardware

The Space originally ran on **ZeroGPU**, whose free quota was exhausted, so visitors saw
*"You have exceeded your ZeroGPU runs limit"* instead of completions. The account owner
switched it to **`CPU basic` (free)**; live generation was then verified end to end in a
browser, with all three models returning completions and reward scores in ~40 s.

The hardware change could not be made with the session's token: both
`POST /api/spaces/.../hardware` and `restart_space` return 401 while file uploads with the
same token succeed, so it lacks the manage-spaces permission.

### Space debugging notes (gradio 6.28 on Spaces)

Three separate defects had to be fixed, all masked behind an empty `{"error": null}`:

1. `gr.Slider` / `gr.Number` raise in `preprocess()` when the browser sends their value as a
   string, which gradio 6.28 does. Replaced with `gr.Dropdown` / `gr.Textbox` and parsed
   in-app.
2. `gr.Examples` sends the example's *text* where its dataset component expects an *index*,
   and because it fires on page load it left the whole UI in an error state so no later
   event reached the backend. Replaced with a plain `gr.Dropdown`.
3. The remaining failure was the ZeroGPU quota above, whose message only appears when the
   request carries the correct `fn_index` (the Examples handler was index 0, `compare` is
   index 1).

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
| mitigation | KL penalty vs frozen base, k3 estimator, `beta` tuned by pilot | brief's default; reward capping considered as a fallback |
| compute | Hugging Face Jobs, `a100-large` flavor | user chose it over Colab; fully CLI-driven |

## Status

| stage | state |
|---|---|
| repo skeleton + package | done |
| local smoke test (gradients flow, KL rises) | done |
| GitHub + HF auth | pending user |
| pilot beta sweep | not started |
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

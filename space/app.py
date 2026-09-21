"""Interactive comparison of the reward-hacked GRPO policy and the KL-regularised one.

Enter (or pick) the opening sentence of a movie review; both policies continue it; the
same sentiment classifier that was used as the training reward scores each continuation.
"""

import json
import os
from pathlib import Path

import gradio as gr
import torch
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer)

try:
    import spaces  # ZeroGPU
    GPU = spaces.GPU
    ZERO = True
except Exception:  # running on CPU hardware or locally
    ZERO = False
    def GPU(*a, **k):
        def deco(fn):
            return fn
        return deco if not a or not callable(a[0]) else a[0]

from packaging.version import Version
from transformers import __version__ as _TV
_DTYPE_KW = "dtype" if Version(_TV).release >= (4, 56) else "torch_dtype"


def _load(cls, name, dtype=None, **kw):
    if dtype is not None:
        kw[_DTYPE_KW] = dtype
    return cls.from_pretrained(name, **kw)


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BASELINE_REPO = "rohanjain2312/grpo-reward-hacked-sentiment-qwen05b"
FIXED_REPO = "rohanjain2312/grpo-reward-hacking-fixed-kl-qwen05b"
REWARD_MODEL = "lvwerra/distilbert-imdb"
GITHUB = "https://github.com/Rohanjain2312/grpo-reward-hacking-demo"

INSTRUCTION = ("Continue the following movie review in 2-3 sentences. "
               "Write only the continuation, in the reviewer's own voice.\n\n{opening}")

EXAMPLES = [
    "I had high hopes for this one, but by the twenty minute mark I was checking my watch.",
    "This has to be one of the laziest scripts I have ever sat through.",
    "The trailer promised a taut thriller and delivered a ninety minute nap.",
    "I wanted to like this film, I really did.",
    "A stellar cast is completely wasted on material this thin.",
    "Every single performance in this movie feels phoned in.",
]

_dtype = torch.float16 if torch.cuda.is_available() or ZERO else torch.float32
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

MODELS = {}
for key, repo in (("base", BASE_MODEL), ("baseline", BASELINE_REPO), ("fixed", FIXED_REPO)):
    try:
        MODELS[key] = _load(AutoModelForCausalLM, repo, dtype=_dtype).eval()
    except Exception as exc:  # a model repo not published yet should not kill the Space
        print(f"[warn] could not load {repo}: {exc}")

rw_tok = AutoTokenizer.from_pretrained(REWARD_MODEL)
rw_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL).eval()
POS = [i for i, l in rw_model.config.id2label.items() if "pos" in str(l).lower()]
POS_IDX = POS[0] if POS else 1

_on_gpu = {"done": False}


def _to_device():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if _on_gpu["done"] and dev == "cpu":
        return dev
    for m in MODELS.values():
        m.to(dev)
    rw_model.to(dev)
    _on_gpu["done"] = True
    return dev


def _reward(texts, dev):
    enc = rw_tok(texts, return_tensors="pt", padding=True, truncation=True,
                 max_length=256).to(dev)
    with torch.no_grad():
        probs = torch.softmax(rw_model(**enc).logits.float(), dim=-1)
    return probs[:, POS_IDX].tolist()


@GPU(duration=90)
def compare(opening, temperature, max_new_tokens, seed):
    opening = (opening or "").strip()
    if not opening:
        return "", "", "Enter a review opening first."
    dev = _to_device()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": INSTRUCTION.format(opening=opening)}],
        tokenize=False, add_generation_prompt=True)
    enc = tokenizer([prompt], return_tensors="pt").to(dev)

    outs = {}
    for key in ("baseline", "fixed"):
        if key not in MODELS:
            outs[key] = "(model repo not available)"
            continue
        torch.manual_seed(int(seed))
        with torch.no_grad():
            gen = MODELS[key].generate(
                **enc, do_sample=True, temperature=float(temperature), top_p=0.95,
                max_new_tokens=int(max_new_tokens), min_new_tokens=8,
                pad_token_id=tokenizer.pad_token_id)
        outs[key] = tokenizer.decode(gen[0][enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()

    texts = [outs["baseline"] or " ", outs["fixed"] or " "]
    r = _reward(texts, dev)
    fmt = lambda t, s: f"**Sentiment reward: {s:.3f}**\n\n{t}"
    note = (f"Both continuations came from the same prompt and seed. The reward is "
            f"P(positive) from `{REWARD_MODEL}` -- the exact signal the baseline was "
            f"trained to maximise.")
    return fmt(outs["baseline"], r[0]), fmt(outs["fixed"], r[1]), note


FIG_DIR = Path(__file__).parent / "figures"
FIGS = [
    ("fig1_reward_hacking_diagnosis.png",
     "The diagnosis: on the baseline run the training reward keeps climbing while an "
     "independent judge's coherence rating falls."),
    ("fig2_reward_vs_step.png", "Reward vs. step, both runs on the same axes."),
    ("fig3_quality_vs_step.png", "Independent quality metric vs. step, both runs."),
    ("fig4_secondary_metrics.png",
     "Perplexity under the frozen base model, bigram diversity, and how far each policy "
     "drifted from the base model in KL."),
    ("fig5_sample_completions.png", "Raw completions side by side at three checkpoints."),
]

INTRO = f"""
# Reward hacking in GRPO, and the one-line fix

Two copies of `Qwen/Qwen2.5-0.5B-Instruct` were trained with **GRPO** to continue the
opening sentence of a negative IMDB review. The reward was `P(positive)` from a real
pretrained sentiment classifier (`lvwerra/distilbert-imdb`) -- a legitimate reward model,
not a keyword heuristic.

* **Baseline** got no KL penalty. It found a degenerate way to satisfy the classifier.
* **Fixed** is the identical run plus a KL penalty against the frozen base policy.

Type a review opening below and watch the two policies diverge.
Code and full write-up: [{GITHUB}]({GITHUB})
"""

with gr.Blocks(title="GRPO reward hacking demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)
    with gr.Tab("Compare the two policies"):
        opening = gr.Textbox(label="Opening sentence of a movie review", lines=2,
                             value=EXAMPLES[0])
        gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[opening])
        with gr.Row():
            temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
            max_new_tokens = gr.Slider(16, 128, value=48, step=8, label="Max new tokens")
            seed = gr.Number(value=12345, precision=0, label="Seed")
        run = gr.Button("Generate both continuations", variant="primary")
        with gr.Row():
            out_base = gr.Markdown(label="Baseline")
            out_fixed = gr.Markdown(label="Fixed")
        with gr.Row():
            gr.Markdown("### Baseline - GRPO, no KL penalty (reward hacked)")
            gr.Markdown("### Fixed - GRPO + KL penalty")
        note = gr.Markdown()
        run.click(compare, [opening, temperature, max_new_tokens, seed],
                  [out_base, out_fixed, note])
    with gr.Tab("Results"):
        for name, caption in FIGS:
            path = FIG_DIR / name
            if path.exists():
                gr.Image(str(path), label=caption, show_label=True,
                         show_download_button=True, container=True)
    with gr.Tab("Method"):
        gr.Markdown(f"""
### What was run

| | |
|---|---|
| base model | `{BASE_MODEL}` |
| reward model | `{REWARD_MODEL}`, reward = P(positive) on the continuation |
| data | `stanfordnlp/imdb`, negative reviews only, first sentence as the prompt |
| algorithm | GRPO: 8 prompts/step x group of 8, group-normalised advantages |
| difference between runs | KL coefficient `beta`: 0.0 (baseline) vs. the fixed run's value |

### Why the KL penalty works

The sentiment classifier is only a *proxy* for "write a positive review". Maximising it
without constraint lets the policy walk to a region of text space where the proxy is
saturated but the text is no longer good writing -- the classifier was never trained on
inputs like that, so its score there is meaningless. The KL term prices every nat of
divergence from the frozen base policy, so a move only happens when the reward gain is
large enough to pay for it. Ordinary improvements in sentiment are cheap; a walk into
degenerate text is not.

### Models

* Reward-hacked: https://huggingface.co/{BASELINE_REPO}
* KL-regularised: https://huggingface.co/{FIXED_REPO}
* Code + write-up: {GITHUB}
""")

if __name__ == "__main__":
    demo.launch()

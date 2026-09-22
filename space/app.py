"""Interactive comparison of the reward-hacked GRPO policy and the KL-regularised one.

Enter (or pick) the opening sentence of a movie review; both policies continue it; the
same sentiment classifier that was used as the training reward scores each continuation.
"""

import json
import os
import traceback
from pathlib import Path

import gradio as gr
import torch
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer)

try:
    import spaces  # ZeroGPU, when the Space is configured for it
    GPU = spaces.GPU
    ZERO = True
except Exception:  # CPU hardware or local: GPU() below is a no-op decorator
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
CAP_REPO = "rohanjain2312/grpo-reward-hacking-fixed-cap-qwen05b"
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

# ZeroGPU forbids initialising CUDA in the main process, so no torch.cuda.*
# calls at import: the dtype is decided from whether we are on ZeroGPU alone.
_dtype = torch.float16 if ZERO else torch.float32
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

MODELS = {}
for key, repo in (("baseline", BASELINE_REPO), ("fixed", FIXED_REPO),
                  ("cap", CAP_REPO)):
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
    """Resolve the device and place the models. Only ever called inside @GPU."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu" and _on_gpu["done"]:
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


ARMS = [("baseline", "Baseline - no constraint (reward hacked)"),
        ("fixed", "Fixed - KL penalty (beta=0.1)"),
        ("cap", "Fixed - reward cap at 0.9")]


@GPU(duration=120)
def _num(value, default, cast=float):
    try:
        return cast(str(value).strip())
    except (TypeError, ValueError):
        return default


def compare(opening, temperature, max_new_tokens, seed):
    opening = (opening or "").strip()
    if not opening:
        return "", "", "", "Enter a review opening first."
    temperature = _num(temperature, 0.7, float)
    max_new_tokens = _num(max_new_tokens, 48, int)
    seed = _num(seed, 12345, int)
    dev = _to_device()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": INSTRUCTION.format(opening=opening)}],
        tokenize=False, add_generation_prompt=True)
    enc = tokenizer([prompt], return_tensors="pt").to(dev)

    outs = {}
    for key, _ in ARMS:
        if key not in MODELS:
            outs[key] = "(model repo not available)"
            continue
        torch.manual_seed(seed)
        with torch.no_grad():
            gen = MODELS[key].generate(
                **enc, do_sample=True, temperature=float(temperature), top_p=0.95,
                max_new_tokens=int(max_new_tokens), min_new_tokens=8,
                pad_token_id=tokenizer.pad_token_id)
        outs[key] = tokenizer.decode(gen[0][enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()

    texts = [outs[k] or " " for k, _ in ARMS]
    r = _reward(texts, dev)
    panes = [f"**Sentiment reward: {s:.3f}**\n\n{t}" for t, s in zip(texts, r)]
    note = (f"All three continuations came from the same prompt and seed. The reward is "
            f"P(positive) from `{REWARD_MODEL}` -- the exact signal the baseline was "
            f"trained to maximise. All three runs reached the same reward (0.996) in "
            f"training; the difference is in the text.")
    return panes[0], panes[1], panes[2], note


FIG_DIR = Path(__file__).parent / "figures"
FIGS = [
    ("fig1_reward_hacking_diagnosis.png",
     "The diagnosis, baseline run. Reward saturates at 0.996 by step 30. Perplexity under "
     "the frozen base model quadruples (4.47 -> 17.96); an independent 7B LLM judge barely "
     "moves (4.32 -> 3.93 of 5). Both are shown, because that gap is itself a finding."),
    ("fig2_reward_vs_step.png",
     "Reward vs. step. All three runs end at 0.996 - neither mitigation cost any reward."),
    ("fig3_quality_vs_step.png",
     "Both independent quality metrics, all three runs. Note the compressed scale on the "
     "judge panel."),
    ("fig4_secondary_metrics.png",
     "The collapse is ACROSS completions, not within them: unique opening phrases fall "
     "0.906 -> 0.547 for the baseline, while per-completion distinct-2 reads ~1.0 the whole "
     "run and sees nothing at all."),
    ("fig5_sample_completions.png",
     "Raw completions at three checkpoints, selected deterministically rather than "
     "hand-picked."),
]

INTRO = f"""
# Reward hacking in GRPO, and two ways to fix it

Three copies of `Qwen/Qwen2.5-0.5B-Instruct` were trained with **GRPO** to continue the
opening sentence of a negative IMDB review. The reward was `P(positive)` from a real
pretrained sentiment classifier (`lvwerra/distilbert-imdb`) -- a legitimate reward model,
not a keyword heuristic.

* **Baseline** got no constraint at all. It found a degenerate way to satisfy the classifier.
* **Fixed (KL)** is the identical run plus a KL penalty against the frozen base policy.
* **Fixed (cap)** instead clips the reward at 0.9, so the group advantage goes to zero once
  everything clears the cap.

All three reached the same reward, **0.996**. Type a review opening below and watch them
diverge anyway.
Code and full write-up: [{GITHUB}]({GITHUB})
"""

with gr.Blocks(title="GRPO reward hacking demo") as demo:
    gr.Markdown(INTRO)
    with gr.Tab("Compare the three policies"):
        opening = gr.Textbox(label="Opening sentence of a movie review", lines=2,
                             value=EXAMPLES[0])
        # A plain Dropdown rather than gr.Examples: on gradio 6 the Examples dataset
        # component sends the example's text where it expects an index, and because it
        # fires on page load it leaves the whole UI in an error state.
        picker = gr.Dropdown(choices=EXAMPLES, value=EXAMPLES[0],
                             label="...or pick an example")
        picker.change(lambda choice: choice, [picker], [opening])
        # Dropdowns/textboxes rather than Slider/Number: gradio 6.28 sends the numeric
        # widgets' values as strings and its own preprocess() then raises on them, so the
        # values are taken as text and parsed in `compare` instead.
        with gr.Row():
            temperature = gr.Dropdown(choices=["0.3", "0.5", "0.7", "0.9", "1.1"],
                                      value="0.7", label="Temperature")
            max_new_tokens = gr.Dropdown(choices=["32", "48", "64", "96", "128"],
                                         value="48", label="Max new tokens")
            seed = gr.Textbox(value="12345", label="Seed")
        run = gr.Button("Generate all three continuations", variant="primary")
        gr.Markdown("*Runs on free CPU hardware - the three generations take "
                    "around 20-40 seconds.*")
        with gr.Row():
            panes = []
            for (key, heading), colour in zip(ARMS, ["#c1272d", "#1f6fb4", "#2e8b57"]):
                with gr.Column():
                    gr.Markdown(f"### <span style='color:{colour}'>{heading}</span>")
                    panes.append(gr.Markdown())
        note = gr.Markdown()
        run.click(compare, [opening, temperature, max_new_tokens, seed], panes + [note])
    with gr.Tab("Results"):
        for name, caption in FIGS:
            path = FIG_DIR / name
            if path.exists():
                gr.Image(str(path), label=caption, show_label=True)
    with gr.Tab("Method"):
        gr.Markdown(f"""
### What was run

| | |
|---|---|
| base model | `{BASE_MODEL}` |
| reward model | `{REWARD_MODEL}`, reward = P(positive) on the continuation |
| data | `stanfordnlp/imdb`, negative reviews only, first sentence as the prompt |
| algorithm | GRPO: 8 prompts/step x group of 8, group-normalised advantages, 100 steps |
| the three arms | no constraint / KL penalty `beta=0.1` / reward clipped at 0.9 |

### Results after 100 steps

| metric | step 0 | baseline | KL | cap |
|---|---|---|---|---|
| sentiment reward (the training signal) | 0.556 | 0.996 | 0.996 | 0.996 |
| perplexity under frozen base model | 4.47 | 17.96 | 6.33 | 8.40 |
| unique opening phrases across eval set | 0.906 | 0.547 | 0.859 | 0.922 |
| LLM-judge coherence (1-5) | 4.32 | 3.93 | 4.17 | 4.03 |
| distinct-2 *within* each completion | 0.997 | 0.999 | 1.000 | 1.000 |

### Why the mitigations work

The classifier is a *proxy*. Maximising it without constraint lets the policy walk to a
region of text space where the proxy is saturated but the text is no longer good writing --
the classifier was never trained on inputs like that, so its score there is meaningless.

The **KL penalty** prices every nat of divergence from the frozen base policy, so a move
only happens when the reward gain pays for it. Ordinary improvements in sentiment are
cheap; a walk into degenerate text is not.

The **reward cap** removes the prize instead: once every sample in a group clears 0.9 their
rewards are identical, so the group-relative advantage is exactly zero and there is no
gradient. Simpler and needs no reference model, but blunter -- it constrains the reward,
not the policy, which is why its perplexity drifts further even though its diversity holds
up better.

### The metrics that missed it

Per-completion distinct-2 reads ~1.0 for every run at every step: the gamed completions are
internally varied, and the repetition is *across* samples. The 7B LLM judge scored the
fully-templated baseline completion 5.00/5 -- the text really is fluent, it is just vacuous,
and a fluency rubric does not price that.

### Models

* Reward-hacked: https://huggingface.co/{BASELINE_REPO}
* KL-regularised: https://huggingface.co/{FIXED_REPO}
* Reward-capped: https://huggingface.co/{CAP_REPO}
* Code + write-up: {GITHUB}
""")

if __name__ == "__main__":
    demo.launch()

"""IMDB prompts: the first sentence of a review is the prompt, the model writes the rest."""

import re

from datasets import load_dataset

_BR = re.compile(r"<br\s*/?>")
_WS = re.compile(r"\s+")
_SENT_END = re.compile(r"(?<=[.!?])\s+")

INSTRUCTION = (
    "Continue the following movie review in 2-3 sentences. "
    "Write only the continuation, in the reviewer's own voice.\n\n{opening}"
)


def _first_sentence(text: str) -> str:
    text = _WS.sub(" ", _BR.sub(" ", text)).strip()
    parts = _SENT_END.split(text, maxsplit=1)
    return parts[0].strip() if parts else ""


def load_prompts(cfg):
    """Return (train_openings, eval_openings) as lists of first-sentence strings.

    Only openings of *negative* reviews are used. The task is therefore "continue this
    negative review" while the reward asks for positive sentiment, which is what creates
    real optimisation pressure -- if the natural continuation already scored ~0.9 there
    would be almost nothing for the policy to gain, and nothing to hack.

    The eval set is disjoint from the training set and is held fixed across runs so the
    quality curves of the baseline and fixed runs are measured on identical prompts.
    """
    ds = load_dataset(cfg.dataset, split="train").shuffle(seed=1234)
    openings, seen = [], set()
    need = cfg.n_train_prompts + cfg.n_eval_prompts
    for row in ds:
        if cfg.label_filter is not None and row["label"] != cfg.label_filter:
            continue
        s = _first_sentence(row["text"])
        if not (cfg.min_prompt_chars <= len(s) <= cfg.max_prompt_chars):
            continue
        if s in seen:
            continue
        seen.add(s)
        openings.append(s)
        if len(openings) >= need:
            break
    if len(openings) < need:
        raise RuntimeError(f"only found {len(openings)} usable openings, need {need}")
    return openings[: cfg.n_train_prompts], openings[cfg.n_train_prompts :]


def build_chat_prompt(tokenizer, opening: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": INSTRUCTION.format(opening=opening)}],
        tokenize=False,
        add_generation_prompt=True,
    )

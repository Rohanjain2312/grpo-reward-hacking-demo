"""Quality metrics that are *not* part of the training signal.

- `base_perplexity`: perplexity of the continuation under the frozen original policy.
  Catches drift into text the original model finds implausible.
- `distinct_2`: fraction of unique word bigrams. Catches repetition loops, which
  perplexity alone does not (a repeated phrase is highly predictable, so degenerate
  repetition can *lower* perplexity while quality collapses).

A third, headline metric -- an LLM-judge coherence rating -- is computed post-hoc by
`grpo_demo.judge` on the completions logged here.
"""

import torch


@torch.no_grad()
def base_perplexity(ref_model, prompt_ids, prompt_mask, completion_ids, completion_mask,
                    micro_batch: int = 8) -> float:
    """Corpus-level perplexity of the completions under the frozen base model."""
    total_nll, total_tokens = 0.0, 0
    n = prompt_ids.size(0)
    for i in range(0, n, micro_batch):
        p_ids = prompt_ids[i : i + micro_batch]
        p_mask = prompt_mask[i : i + micro_batch]
        c_ids = completion_ids[i : i + micro_batch]
        c_mask = completion_mask[i : i + micro_batch]
        ids = torch.cat([p_ids, c_ids], dim=1)
        mask = torch.cat([p_mask, c_mask], dim=1)
        position_ids = (mask.cumsum(dim=-1) - 1).clamp(min=0)
        logits = ref_model(input_ids=ids, attention_mask=mask, position_ids=position_ids).logits
        logits = logits[:, :-1, :]
        labels = ids[:, 1:]
        k = c_ids.size(1)
        logits, labels = logits[:, -k:, :], labels[:, -k:]
        logp = torch.log_softmax(logits.float(), dim=-1)
        token_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        total_nll += float((-token_logp * c_mask).sum())
        total_tokens += int(c_mask.sum())
    if total_tokens == 0:
        return float("nan")
    return float(torch.exp(torch.tensor(total_nll / total_tokens)))


def distinct_2(texts) -> float:
    """Mean per-completion ratio of unique bigrams to total bigrams."""
    ratios = []
    for t in texts:
        words = t.split()
        if len(words) < 2:
            ratios.append(0.0)
            continue
        bigrams = list(zip(words[:-1], words[1:]))
        ratios.append(len(set(bigrams)) / len(bigrams))
    return float(sum(ratios) / len(ratios)) if ratios else float("nan")


def cross_sample_diversity(texts, prefix_words: int = 4) -> dict:
    """Diversity *across* the eval set, not within a single completion.

    Added after observing the actual failure mode: the gamed policy converges on one
    high-scoring opening template ("Absolutely captivating and deeply ...") while each
    individual completion stays internally varied. Per-completion `distinct_2` is ~1.0
    throughout that collapse, because it never compares one sample to another.

    Computed post-hoc from logged samples, so it needs no retraining.
    """
    texts = [t for t in texts if t.strip()]
    if not texts:
        return {"corpus_distinct_2": float("nan"), "unique_prefix_ratio": float("nan")}

    pooled = []
    for t in texts:
        w = t.split()
        pooled.extend(zip(w[:-1], w[1:]))
    corpus_d2 = len(set(pooled)) / len(pooled) if pooled else float("nan")

    prefixes = [" ".join(t.split()[:prefix_words]).lower() for t in texts]
    return {
        "corpus_distinct_2": float(corpus_d2),
        "unique_prefix_ratio": len(set(prefixes)) / len(prefixes),
    }

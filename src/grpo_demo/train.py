"""GRPO training loop for the reward-hacking demo.

Deliberately a small, readable custom loop rather than a framework trainer: the article
this demo accompanies explains GRPO step by step, and the KL term is the one line the
whole experiment turns on.

    python -m grpo_demo.train --config baseline
    python -m grpo_demo.train --config fixed_kl
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from .config import get_config
from .data import build_chat_prompt, load_prompts
from .hub import METRICS, SAMPLES, HubCheckpointer, append_jsonl, rewrite_jsonl
from .quality import base_perplexity, distinct_2
from .rewards import SentimentReward


# ----------------------------------------------------------------- utilities
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


def selective_log_softmax(logits, index):
    """Per-token log-probs without materialising a full float32 log-softmax."""
    rows = []
    for row_logits, row_index in zip(logits, index):
        logp = torch.log_softmax(row_logits.float(), dim=-1)
        rows.append(logp.gather(-1, row_index.unsqueeze(-1)).squeeze(-1))
    return torch.stack(rows)


def token_logps(model, prompt_ids, prompt_mask, completion_ids, completion_mask,
                requires_grad: bool):
    """Log-probs of the completion tokens under `model`, shape (N, T_completion)."""
    ids = torch.cat([prompt_ids, completion_ids], dim=1)
    mask = torch.cat([prompt_mask, completion_mask], dim=1)
    position_ids = (mask.cumsum(dim=-1) - 1).clamp(min=0)
    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        logits = model(input_ids=ids, attention_mask=mask,
                       position_ids=position_ids, use_cache=False).logits
        logits = logits[:, :-1, :]
        labels = ids[:, 1:]
        k = completion_ids.size(1)
        return selective_log_softmax(logits[:, -k:, :], labels[:, -k:])


def completion_mask_from(completion_ids, eos_ids):
    """1 up to and including the first EOS, 0 afterwards."""
    is_eos = torch.zeros_like(completion_ids, dtype=torch.bool)
    for e in eos_ids:
        is_eos |= completion_ids == e
    n, t = completion_ids.shape
    first_eos = torch.full((n,), t, dtype=torch.long, device=completion_ids.device)
    has_eos = is_eos.any(dim=1)
    first_eos[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    idx = torch.arange(t, device=completion_ids.device).expand(n, t)
    return (idx <= first_eos.unsqueeze(1)).long()


@torch.no_grad()
def generate(model, tokenizer, prompts, device, *, max_new_tokens, temperature, top_p,
             num_return_sequences=1, micro_batch=16):
    """Sample completions. Returns (prompt_ids, prompt_mask, completion_ids, completion_mask)."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=192).to(device)
    p_ids, p_mask = enc["input_ids"], enc["attention_mask"]
    eos_ids = [i for i in {tokenizer.eos_token_id, tokenizer.pad_token_id} if i is not None]

    out_p_ids, out_p_mask, out_c_ids = [], [], []
    for i in range(0, p_ids.size(0), micro_batch):
        chunk_ids, chunk_mask = p_ids[i : i + micro_batch], p_mask[i : i + micro_batch]
        gen = model.generate(
            input_ids=chunk_ids,
            attention_mask=chunk_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=0,
            max_new_tokens=max_new_tokens,
            min_new_tokens=4,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
        )
        completions = gen[:, chunk_ids.size(1) :]
        # pad every chunk to the same completion length so chunks can be concatenated
        if completions.size(1) < max_new_tokens:
            pad = torch.full(
                (completions.size(0), max_new_tokens - completions.size(1)),
                tokenizer.pad_token_id, dtype=completions.dtype, device=completions.device)
            completions = torch.cat([completions, pad], dim=1)
        out_c_ids.append(completions)
        out_p_ids.append(chunk_ids.repeat_interleave(num_return_sequences, dim=0))
        out_p_mask.append(chunk_mask.repeat_interleave(num_return_sequences, dim=0))

    c_ids = torch.cat(out_c_ids, dim=0)
    return (torch.cat(out_p_ids, 0), torch.cat(out_p_mask, 0), c_ids,
            completion_mask_from(c_ids, eos_ids))


def decode(tokenizer, completion_ids, completion_mask):
    texts = []
    for ids, mask in zip(completion_ids, completion_mask):
        kept = ids[mask.bool()]
        texts.append(tokenizer.decode(kept, skip_special_tokens=True).strip())
    return texts


# --------------------------------------------------------------------- eval
def evaluate(policy, ref_model, tokenizer, reward_fn, eval_prompts, eval_openings, cfg,
             device, step):
    torch.manual_seed(12345)
    policy.eval()
    p_ids, p_mask, c_ids, c_mask = generate(
        policy, tokenizer, eval_prompts, device,
        max_new_tokens=cfg.eval_max_new_tokens,
        temperature=cfg.eval_temperature, top_p=cfg.eval_top_p,
        micro_batch=cfg.micro_batch,
    )
    policy.train()
    texts = decode(tokenizer, c_ids, c_mask)
    rewards = reward_fn.score(texts)
    ppl = base_perplexity(ref_model, p_ids, p_mask, c_ids, c_mask,
                          micro_batch=max(2, cfg.micro_batch // 2))
    d2 = distinct_2(texts)
    lengths = c_mask.sum(dim=1).float()

    samples = [
        {"step": step, "opening": o, "completion": t, "reward": float(r),
         "n_tokens": int(l)}
        for o, t, r, l in zip(eval_openings, texts, rewards.tolist(), lengths.tolist())
    ]
    metrics = {
        "eval_reward": float(rewards.mean()),
        "eval_perplexity": float(ppl),
        "eval_distinct_2": float(d2),
        "eval_completion_tokens": float(lengths.mean()),
    }
    return metrics, samples


# --------------------------------------------------------------------- main
def train(cfg, push: bool = True):
    device, dtype = pick_device()
    print(f"[setup] device={device} dtype={dtype} run={cfg.run} beta={cfg.beta}", flush=True)

    out_dir = Path(cfg.output_dir) / cfg.run
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path, samples_path = out_dir / "metrics.jsonl", out_dir / "samples.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(cfg.policy_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ckpt = HubCheckpointer(cfg.hf_repo, local_dir=str(out_dir / "hub"), enabled=push)
    latest = ckpt.find_latest()

    load_from = cfg.policy_model
    start_step, opt_state = 0, None
    if latest is not None:
        local = ckpt.download_checkpoint(latest["checkpoint"])
        load_from = str(local)
        start_step = int(latest["step"])
        if latest.get("has_optimizer") and (local / "optimizer.pt").exists():
            opt_state = torch.load(local / "optimizer.pt", map_location="cpu")
        prev_metrics, prev_samples = ckpt.download_logs()
        rewrite_jsonl(metrics_path, [m for m in prev_metrics if m["step"] <= start_step])
        rewrite_jsonl(samples_path, [s for s in prev_samples if s["step"] <= start_step])
        print(f"[resume] found checkpoint at step {start_step} -> resuming", flush=True)
    else:
        metrics_path.unlink(missing_ok=True)
        samples_path.unlink(missing_ok=True)
        print("[resume] no checkpoint found -> starting from step 0", flush=True)

    policy = AutoModelForCausalLM.from_pretrained(load_from, dtype=dtype).to(device)
    policy.config.use_cache = True
    policy.train()
    ref_model = AutoModelForCausalLM.from_pretrained(cfg.policy_model, dtype=dtype).to(device).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    reward_fn = SentimentReward(cfg.reward_model, device)

    train_openings, eval_openings = load_prompts(cfg)
    train_prompts = [build_chat_prompt(tokenizer, o) for o in train_openings]
    eval_prompts = [build_chat_prompt(tokenizer, o) for o in eval_openings]

    order = np.random.RandomState(cfg.seed).permutation(len(train_prompts))

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate,
                                  betas=(0.9, 0.95), weight_decay=0.0)
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)
    scheduler = get_linear_schedule_with_warmup(optimizer, cfg.warmup_steps, cfg.total_steps)
    for _ in range(start_step):
        scheduler.step()

    if start_step == 0:
        m, s = evaluate(policy, ref_model, tokenizer, reward_fn, eval_prompts,
                        eval_openings, cfg, device, 0)
        append_jsonl(metrics_path, {"step": 0, **m})
        for rec in s:
            append_jsonl(samples_path, rec)
        print(f"[eval] step 0 {json.dumps(m)}", flush=True)
        ckpt.push_logs(metrics_path, samples_path, 0)

    t0 = time.time()
    for step in range(start_step + 1, cfg.total_steps + 1):
        torch.manual_seed(cfg.seed * 100003 + step)
        base = (step - 1) * cfg.prompts_per_step
        idx = [int(order[(base + j) % len(order)]) for j in range(cfg.prompts_per_step)]
        batch_prompts = [train_prompts[i] for i in idx]

        policy.eval()
        p_ids, p_mask, c_ids, c_mask = generate(
            policy, tokenizer, batch_prompts, device,
            max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature,
            top_p=cfg.top_p, num_return_sequences=cfg.group_size,
            micro_batch=max(1, cfg.micro_batch // cfg.group_size),
        )
        policy.train()

        texts = decode(tokenizer, c_ids, c_mask)
        rewards = reward_fn.score(texts).to(device)

        grouped = rewards.view(cfg.prompts_per_step, cfg.group_size)
        mean = grouped.mean(dim=1, keepdim=True)
        std = grouped.std(dim=1, keepdim=True)
        advantages = ((grouped - mean) / (std + cfg.adv_eps)).view(-1)

        total_tokens = c_mask.sum().clamp(min=1)
        optimizer.zero_grad(set_to_none=True)
        kl_sum, pg_sum = 0.0, 0.0
        n = p_ids.size(0)
        for i in range(0, n, cfg.micro_batch):
            sl = slice(i, i + cfg.micro_batch)
            mb_mask = c_mask[sl]
            logps = token_logps(policy, p_ids[sl], p_mask[sl], c_ids[sl], mb_mask,
                                requires_grad=True)
            with torch.no_grad():
                ref_logps = token_logps(ref_model, p_ids[sl], p_mask[sl], c_ids[sl], mb_mask,
                                        requires_grad=False)
            # k3 estimator (Schulman): unbiased, non-negative
            delta = ref_logps - logps
            kl = torch.exp(delta) - delta - 1.0
            pg = -advantages[sl].unsqueeze(1) * logps
            per_token = pg + cfg.beta * kl
            loss = (per_token * mb_mask).sum() / total_tokens
            loss.backward()
            kl_sum += float((kl.detach() * mb_mask).sum())
            pg_sum += float((pg.detach() * mb_mask).sum())

        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()

        denom = float(total_tokens)
        row = {
            "step": step,
            "train_reward": float(rewards.mean()),
            "train_reward_std": float(rewards.std()),
            "kl": kl_sum / denom,
            "pg_loss": pg_sum / denom,
            "grad_norm": float(grad_norm),
            "lr": scheduler.get_last_lr()[0],
            "completion_tokens": float(c_mask.sum(dim=1).float().mean()),
            "elapsed_s": round(time.time() - t0, 1),
        }

        if step % cfg.eval_every == 0 or step == cfg.total_steps:
            m, s = evaluate(policy, ref_model, tokenizer, reward_fn, eval_prompts,
                            eval_openings, cfg, device, step)
            row.update(m)
            for rec in s:
                append_jsonl(samples_path, rec)
            print(f"[eval] step {step} {json.dumps(m)}", flush=True)

        append_jsonl(metrics_path, row)
        print(f"[step {step}/{cfg.total_steps}] reward={row['train_reward']:.4f} "
              f"kl={row['kl']:.4f} gn={row['grad_norm']:.2f} "
              f"t={row['elapsed_s']}s", flush=True)

        if step % cfg.eval_every == 0:
            ckpt.push_logs(metrics_path, samples_path, step)
        if step % cfg.full_ckpt_every == 0 and step != cfg.total_steps:
            ckpt.push_checkpoint(step, policy, tokenizer, optimizer,
                                 {"step": step, "config": cfg.to_dict()},
                                 metrics_path, samples_path, with_optimizer=True)
            print(f"[ckpt] pushed full checkpoint @ step {step}", flush=True)

    ckpt.push_checkpoint(cfg.total_steps, policy, tokenizer, optimizer,
                         {"step": cfg.total_steps, "config": cfg.to_dict()},
                         metrics_path, samples_path, with_optimizer=True)
    from .cards import model_card
    ckpt.push_final(policy, tokenizer, model_card(cfg, metrics_path), metrics_path, samples_path)
    print(f"[done] {cfg.run} finished in {round(time.time() - t0, 1)}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=["baseline", "fixed_kl"])
    ap.add_argument("--total-steps", type=int)
    ap.add_argument("--beta", type=float)
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--prompts-per-step", type=int)
    ap.add_argument("--group-size", type=int)
    ap.add_argument("--eval-every", type=int)
    ap.add_argument("--n-eval-prompts", type=int)
    ap.add_argument("--micro-batch", type=int)
    ap.add_argument("--policy-model", type=str)
    ap.add_argument("--n-train-prompts", type=int)
    ap.add_argument("--max-new-tokens", type=int)
    ap.add_argument("--full-ckpt-every", type=int)
    ap.add_argument("--hf-repo", type=str)
    ap.add_argument("--output-dir", type=str)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()
    overrides = {k: v for k, v in vars(args).items()
                 if k not in {"config", "no_push"} and v is not None}
    cfg = get_config(args.config, **overrides)
    train(cfg, push=not args.no_push)


if __name__ == "__main__":
    main()

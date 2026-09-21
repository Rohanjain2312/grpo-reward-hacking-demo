"""LLM-judge coherence scoring -- the headline quality metric.

A stronger open model (Qwen2.5-7B-Instruct) rates each logged completion 1-5 on writing
quality against a fixed rubric. Rather than sampling a digit, we read the probability the
judge assigns to each of the tokens "1".."5" at the answer position and take the
expectation, which is deterministic and gives a smooth curve.

The rubric deliberately tells the judge to ignore sentiment, so the judge score is
independent of the training reward.

    python -m grpo_demo.judge --runs baseline fixed_kl
"""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from .compat import load
from .config import CONFIGS

JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

RUBRIC = """You are grading the writing quality of a continuation of a movie review.

Review opening:
{opening}

Continuation to grade:
{completion}

Rate ONLY the continuation's writing quality on a 1-5 scale. Judge fluency, coherence and
whether it reads like real prose from a movie review. Do NOT reward or penalise the
opinion expressed: a well-written negative review and a well-written positive review both
score 5.

1 = degenerate: repetition loops, word salad, or gibberish
2 = mostly broken, only fragments are readable
3 = readable but clearly flawed: awkward, repetitive or off-topic
4 = good: fluent and on-topic with minor problems
5 = excellent: fluent, coherent, natural review prose

Answer with a single digit from 1 to 5 and nothing else."""


class CoherenceJudge:
    def __init__(self, model_name=JUDGE_MODEL, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = load(AutoModelForCausalLM, model_name,
                          dtype=dtype).to(self.device).eval()
        self.digit_ids = [self.tok.encode(str(d), add_special_tokens=False)[0]
                          for d in range(1, 6)]
        self.values = torch.arange(1, 6, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def score(self, pairs, batch_size: int = 16):
        """pairs: list of (opening, completion). Returns list of floats in [1, 5]."""
        out = []
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            texts = [
                self.tok.apply_chat_template(
                    [{"role": "user",
                      "content": RUBRIC.format(opening=o, completion=c if c.strip() else "(empty)")}],
                    tokenize=False, add_generation_prompt=True)
                for o, c in chunk
            ]
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                           max_length=1024).to(self.device)
            logits = self.model(**enc).logits[:, -1, :].float()
            probs = torch.softmax(logits[:, self.digit_ids], dim=-1)
            out.extend((probs * self.values).sum(dim=-1).tolist())
        return out


def _load_samples(run: str, cfg, results_dir: Path):
    local = results_dir / run / "samples.jsonl"
    if local.exists():
        with open(local) as f:
            return [json.loads(l) for l in f if l.strip()]
    path = hf_hub_download(cfg.hf_repo, "logs/samples.jsonl", repo_type="model")
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["baseline", "fixed_kl"])
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    judge = CoherenceJudge(args.judge_model)
    api = HfApi()

    for run in args.runs:
        cfg = CONFIGS[run]
        samples = _load_samples(run, cfg, results_dir)
        scores = judge.score([(s["opening"], s["completion"]) for s in samples],
                             batch_size=args.batch_size)
        out_path = results_dir / run / "judge.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for s, sc in zip(samples, scores):
                f.write(json.dumps({"step": s["step"], "opening": s["opening"],
                                    "completion": s["completion"], "reward": s["reward"],
                                    "judge_score": sc}) + "\n")
        by_step = {}
        for s, sc in zip(samples, scores):
            by_step.setdefault(s["step"], []).append(sc)
        for step in sorted(by_step):
            print(f"[{run}] step {step:4d} judge={sum(by_step[step])/len(by_step[step]):.3f}",
                  flush=True)
        if not args.no_push:
            api.upload_file(path_or_fileobj=str(out_path), path_in_repo="logs/judge.jsonl",
                            repo_id=cfg.hf_repo, repo_type="model",
                            commit_message="LLM-judge coherence scores")


if __name__ == "__main__":
    main()

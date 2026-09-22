"""Publication-quality figures for the README, the Space and the article.

    python -m grpo_demo.figures
"""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .config import CONFIGS
from .quality import cross_sample_diversity

ALL_RUNS = ["baseline", "fixed_kl", "fixed_cap"]
LABEL = {"baseline": "Baseline (no KL) - gamed",
         "fixed_kl": "Fixed (KL penalty, beta=0.1)",
         "fixed_cap": "Fixed (reward cap at 0.9)"}
COLOR = {"baseline": "#c1272d", "fixed_kl": "#1f6fb4", "fixed_cap": "#2e8b57"}
MARKER = {"baseline": "o", "fixed_kl": "s", "fixed_cap": "^"}

# Only runs that actually have results are plotted, so the reward-cap arm appears
# automatically once it has been run and is silently absent until then.
RUNS = list(ALL_RUNS)

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# ------------------------------------------------------------------- loading
def _read_jsonl(path: Path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_run(run: str, results_dir: Path):
    metrics = _read_jsonl(results_dir / run / "metrics.jsonl")
    judge = _read_jsonl(results_dir / run / "judge.jsonl")
    evals = [m for m in metrics if "eval_reward" in m]
    judge_by_step = {}
    for r in judge:
        judge_by_step.setdefault(r["step"], []).append(r["judge_score"])
    judge_mean = {s: sum(v) / len(v) for s, v in judge_by_step.items()}
    samples = judge or _read_jsonl(results_dir / run / "samples.jsonl")

    # Cross-sample diversity is computed here rather than during training: the failure
    # mode it detects (one shared template across completions) only became apparent
    # after the fact, and it is recoverable from the logged samples.
    by_step = {}
    for r in samples:
        by_step.setdefault(r["step"], []).append(r["completion"])
    diversity = {st: cross_sample_diversity(v) for st, v in by_step.items()}
    for e in evals:
        d = diversity.get(e["step"])
        if d:
            e.update(d)

    return {"metrics": metrics, "evals": evals, "judge": judge_mean,
            "samples": samples, "diversity": diversity}


def _series(evals, key):
    xs = [e["step"] for e in evals if e.get(key) is not None]
    ys = [e[key] for e in evals if e.get(key) is not None]
    return xs, ys


# ------------------------------------------------------------------- figures
def _twin(ax, steps, reward, q_steps, q_vals, q_axis, q_label, q_lim, title):
    ax.plot(steps, reward, color="#c1272d", marker="o", lw=2.2, ms=5,
            label="Sentiment reward (the training signal)")
    ax.set_xlabel("GRPO training step")
    ax.set_ylabel("Reward: P(positive)", color="#c1272d")
    ax.tick_params(axis="y", colors="#c1272d")
    ax.set_ylim(0, 1.05)
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(q_steps, q_vals, color="#2a2a2a", marker="s", lw=2.2, ms=5, ls="--",
             label=q_label)
    ax2.set_ylabel(q_axis, color="#2a2a2a")
    ax2.set_ylim(*q_lim)
    ax2.grid(False)
    ax.set_title(title, fontsize=11.5)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=8.5)


def fig_diagnosis(data, out: Path):
    """Reward up, quality down, on the baseline run -- shown against BOTH quality metrics.

    Both panels are plotted rather than only the more dramatic one: perplexity moves ~4x,
    the LLM judge barely 0.4 of its 5 points, and that gap is itself a finding.
    """
    d = data["baseline"]
    steps, reward = _series(d["evals"], "eval_reward")
    p_steps, ppl = _series(d["evals"], "eval_perplexity")
    has_judge = bool(d["judge"])

    ncols = 2 if has_judge else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7.8 * ncols, 4.6))
    axes = axes if has_judge else [axes]

    _twin(axes[0], steps, reward, p_steps, ppl,
          "Perplexity (higher = worse)",
          "Perplexity under frozen base model",
          (0, max(ppl) * 1.15),
          "Against perplexity: a ~4x collapse")
    if has_judge:
        j_steps = sorted(d["judge"])
        _twin(axes[1], steps, reward, j_steps, [d["judge"][s] for s in j_steps],
              "Judge coherence (1-5)",
              "Coherence, independent 7B LLM judge",
              (1, 5.1),
              "Against an LLM judge: real, but far smaller")

    fig.suptitle("Reward hacking: the reward saturates while quality degrades",
                 fontsize=13.5, fontweight="bold", y=1.0)
    fig.text(0.5, -0.04, "GRPO on IMDB negative-review continuation, Qwen2.5-0.5B-Instruct, "
             "reward = P(positive) from lvwerra/distilbert-imdb, no KL penalty",
             ha="center", fontsize=8.5, color="#555")
    fig.subplots_adjust(wspace=0.38)
    fig.savefig(out / "fig1_reward_hacking_diagnosis.png")
    plt.close(fig)


def fig_reward(data, out: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for run in RUNS:
        steps, reward = _series(data[run]["evals"], "eval_reward")
        if not steps:
            continue
        ax.plot(steps, reward, color=COLOR[run], marker=MARKER[run], lw=2.2, ms=5,
                label=LABEL[run])
    ax.set_xlabel("GRPO training step")
    ax.set_ylabel("Sentiment reward: P(positive)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Reward vs. training step")
    ax.legend(loc="lower right", fontsize=10)
    fig.text(0.5, -0.04, "Held-out eval prompts, identical across both runs",
             ha="center", fontsize=8.5, color="#555")
    fig.savefig(out / "fig2_reward_vs_step.png")
    plt.close(fig)


def fig_quality(data, out: Path):
    """Both independent quality metrics, every arm, on comparable axes."""
    has_judge = any(data[r]["judge"] for r in RUNS)
    ncols = 2 if has_judge else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7.6 * ncols, 4.4))
    axes = axes if has_judge else [axes]

    for run in RUNS:
        steps, ppl = _series(data[run]["evals"], "eval_perplexity")
        axes[0].plot(steps, ppl, color=COLOR[run], marker=MARKER[run], lw=2.2, ms=5,
                     label=LABEL[run])
    axes[0].set_ylabel("Perplexity under frozen base model")
    axes[0].set_title("Perplexity (higher = worse)", fontsize=11.5)
    axes[0].legend(loc="upper left", fontsize=9)

    if has_judge:
        for run in RUNS:
            js = sorted(data[run]["judge"])
            if not js:
                continue
            axes[1].plot(js, [data[run]["judge"][s] for s in js], color=COLOR[run],
                         marker=MARKER[run], lw=2.2, ms=5, label=LABEL[run])
        axes[1].set_ylabel("Judge coherence score (1-5)")
        axes[1].set_ylim(3.0, 5.0)
        axes[1].set_title("LLM judge (higher = better) - note the compressed scale",
                          fontsize=11.5)
        axes[1].legend(loc="lower left", fontsize=9)

    for ax in axes:
        ax.set_xlabel("GRPO training step")
    fig.suptitle("Output quality vs. training step - neither metric is part of the "
                 "training signal", fontsize=13, fontweight="bold", y=1.01)
    fig.subplots_adjust(wspace=0.26)
    fig.savefig(out / "fig3_quality_vs_step.png")
    plt.close(fig)


def fig_secondary(data, out: Path):
    panels = [
        ("unique_prefix_ratio", "Unique opening phrases\nACROSS the eval set",
         "catches it: lower = one template reused"),
        ("corpus_distinct_2", "Distinct-2 pooled\nACROSS the eval set",
         "catches it: lower = shared phrasing"),
        ("eval_distinct_2", "Distinct-2 WITHIN\neach completion",
         "misses it: flat at ~1.0 all run"),
        ("kl", "KL(policy || frozen base)\nper token",
         "how far the policy drifted"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.3))
    handles = None
    for ax, (key, title, sub) in zip(axes, panels):
        for run in RUNS:
            source = data[run]["metrics"] if key == "kl" else data[run]["evals"]
            steps, vals = _series(source, key)
            if not steps:
                continue
            if key == "kl":  # per-step training metric: smooth for legibility
                w = 5
                vals = [sum(vals[max(0, i - w + 1): i + 1]) / len(vals[max(0, i - w + 1): i + 1])
                        for i in range(len(vals))]
                ax.plot(steps, vals, color=COLOR[run], lw=2.0, label=LABEL[run])
            else:
                ax.plot(steps, vals, color=COLOR[run], marker=MARKER[run], lw=2.0, ms=4,
                        label=LABEL[run])
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("GRPO training step", fontsize=9.5)
        ax.tick_params(labelsize=9)
        if key == "kl":
            ax.set_yscale("symlog", linthresh=1e-3)
        else:
            ax.set_ylim(0, 1.08)
        ax.text(0.5, -0.30, sub, transform=ax.transAxes, fontsize=8.5, color="#666",
                ha="center", va="top", style="italic")
        if handles is None:
            handles = ax.get_lines()[:2]
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center",
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("The collapse is ACROSS completions, not within them - so the usual "
                 "within-sample diversity metric never sees it",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.subplots_adjust(bottom=0.26, wspace=0.28)
    fig.savefig(out / "fig4_secondary_metrics.png")
    plt.close(fig)


def _pick_samples(records, step, n=2):
    rows = [r for r in records if r["step"] == step]
    rows.sort(key=lambda r: r["opening"])  # deterministic, not cherry-picked
    return rows[:n]


def fig_samples(data, out: Path, steps=None, per_step=2):
    """Side-by-side raw completions with their reward and judge score.

    Row heights are derived from the wrapped text so cells never overflow into the row
    below, which they do at four columns with a fixed row height.
    """
    base = data["baseline"]
    all_steps = sorted(base["judge"] or {r["step"] for r in base["samples"]})
    if steps is None:
        steps = [all_steps[0], all_steps[len(all_steps) // 2], all_steps[-1]] if all_steps else []
    if not steps:
        return

    n_cols = len(RUNS) + 1
    wrap_at = max(26, int(140 / n_cols))
    char_cap = max(180, int(960 / n_cols))

    # lay the content out first so row heights can follow it
    layout = []
    for step in steps:
        picked = {r: _pick_samples(data[r]["samples"], step, per_step) for r in RUNS}
        for i in range(per_step):
            rows = [picked[r][i] if i < len(picked[r]) else None for r in RUNS]
            if not any(rows):
                continue
            opening = next(r["opening"] for r in rows if r)
            cells = [textwrap.fill(opening, wrap_at)]
            for rec in rows:
                cells.append("" if rec is None
                             else textwrap.fill(rec["completion"][:char_cap] or "(empty)", wrap_at))
            n_lines = max(c.count("\n") + 1 for c in cells)
            layout.append({"step": step, "first": i == 0, "cells": cells,
                           "recs": rows, "lines": n_lines})

    LINE_IN = 0.135          # inches per wrapped line
    PAD_IN = 0.46            # score line + padding
    HEADER_IN = 0.46
    TITLE_IN = 0.52
    heights = [r["lines"] * LINE_IN + PAD_IN + (0.20 if r["first"] else 0) for r in layout]
    fig_h = TITLE_IN + HEADER_IN + sum(heights)
    fig_w = 4.2 * n_cols

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    gap = 0.008
    col_w = (1.0 - gap * n_cols) / n_cols
    col_x = [0.004 + i * (col_w + gap) for i in range(n_cols)]
    headers = ["Review opening (prompt)"] + [LABEL[r] for r in RUNS]
    header_c = ["#2a2a2a"] + [COLOR[r] for r in RUNS]

    ttl_h = TITLE_IN / fig_h
    hdr_h = HEADER_IN / fig_h
    band_top = 1 - ttl_h - hdr_h * 0.85
    for x, h, c in zip(col_x, headers, header_c):
        ax.add_patch(Rectangle((x, band_top), col_w, hdr_h * 0.72, color=c,
                               alpha=0.12, transform=ax.transAxes))
        ax.text(x + 0.006, band_top + hdr_h * 0.36, h, fontsize=10.5, fontweight="bold",
                color=c, va="center", transform=ax.transAxes)

    y = 1 - ttl_h - hdr_h
    for row, h_in in zip(layout, heights):
        h = h_in / fig_h
        top = y
        y -= h
        ax.plot([0, 1], [y, y], color="#dddddd", lw=0.8, transform=ax.transAxes)
        off = 0.0
        if row["first"]:
            ax.text(0.004, top - 0.012, f"step {row['step']}", fontsize=9.5,
                    fontweight="bold", color="#888", transform=ax.transAxes)
            off = 0.20 / fig_h
        ax.text(col_x[0] + 0.006, top - off - 0.004, row["cells"][0], fontsize=8.6,
                va="top", color="#333", transform=ax.transAxes, style="italic")
        for j, rec in enumerate(row["recs"], start=1):
            if rec is None:
                continue
            score = f"reward {rec['reward']:.2f}"
            if "judge_score" in rec:
                score += f"   |   judge {rec['judge_score']:.2f}/5"
            ax.text(col_x[j] + 0.006, top - off - 0.004, score, fontsize=8.4,
                    fontweight="bold", color=header_c[j], va="top", transform=ax.transAxes)
            ax.text(col_x[j] + 0.006, top - off - 0.004 - (0.22 / fig_h), row["cells"][j],
                    fontsize=8.3, va="top", color="#222", transform=ax.transAxes)

    ax.text(0.5, 1 - ttl_h * 0.48, "Same prompts, same step: gamed vs. mitigated completions",
            fontsize=13, fontweight="bold", ha="center", va="center",
            transform=ax.transAxes)
    fig.savefig(out / "fig5_sample_completions.png")
    plt.close(fig)


def samples_markdown(data, results_dir: Path, steps=None, per_step=2) -> str:
    all_steps = sorted({r["step"] for r in data["baseline"]["samples"]})
    if steps is None:
        steps = [all_steps[0], all_steps[len(all_steps) // 2], all_steps[-1]]

    def cell(r):
        if r is None:
            return ""
        txt = r["completion"].replace("|", "\\|").replace("\n", " ")[:300]
        tag = f"**reward {r['reward']:.2f}"
        tag += f" / judge {r['judge_score']:.2f}**" if "judge_score" in r else "**"
        return f"{tag}<br>{txt}"

    lines = []
    for step in steps:
        lines.append(f"\n#### Step {step}\n")
        lines.append("| | " + " | ".join(LABEL[r] for r in RUNS) + " |")
        lines.append("|" + "---|" * (len(RUNS) + 1))
        picked = {r: _pick_samples(data[r]["samples"], step, per_step) for r in RUNS}
        for i in range(per_step):
            rows = [picked[r][i] if i < len(picked[r]) else None for r in RUNS]
            if not any(rows):
                continue
            opening = next(r["opening"] for r in rows if r).replace("|", "\\|")
            lines.append(f"| *{opening}* | " + " | ".join(cell(r) for r in rows) + " |")
    return "\n".join(lines)


def main():
    global RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    results_dir, out = Path(args.results_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    RUNS = [r for r in ALL_RUNS if (results_dir / r / "metrics.jsonl").exists()]
    print("runs found:", ", ".join(RUNS))
    data = {run: load_run(run, results_dir) for run in RUNS}

    fig_diagnosis(data, out)
    fig_reward(data, out)
    fig_quality(data, out)
    fig_secondary(data, out)
    fig_samples(data, out)
    md = samples_markdown(data, results_dir)
    Path("docs").mkdir(exist_ok=True)
    (Path("docs") / "sample_completions.md").write_text(md)
    print("wrote:", *sorted(p.name for p in out.glob("*.png")), sep="\n  ")


if __name__ == "__main__":
    main()

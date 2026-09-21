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

RUNS = ["baseline", "fixed_kl"]
LABEL = {"baseline": "Baseline (no KL) - gamed", "fixed_kl": "Fixed (KL penalty, beta=0.1)"}
COLOR = {"baseline": "#c1272d", "fixed_kl": "#1f6fb4"}
MARKER = {"baseline": "o", "fixed_kl": "s"}

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
    return {"metrics": metrics, "evals": evals, "judge": judge_mean,
            "samples": judge or _read_jsonl(results_dir / run / "samples.jsonl")}


def _series(evals, key):
    xs = [e["step"] for e in evals if e.get(key) is not None]
    ys = [e[key] for e in evals if e.get(key) is not None]
    return xs, ys


# ------------------------------------------------------------------- figures
def fig_diagnosis(data, out: Path):
    """The money shot: reward up, quality down, same axis, baseline run only."""
    d = data["baseline"]
    steps, reward = _series(d["evals"], "eval_reward")
    j_steps = sorted(d["judge"])
    j_vals = [d["judge"][s] for s in j_steps]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(steps, reward, color="#c1272d", marker="o", lw=2.2, ms=5,
            label="Sentiment reward (training signal)")
    ax.set_xlabel("GRPO training step")
    ax.set_ylabel("Reward: P(positive)", color="#c1272d")
    ax.tick_params(axis="y", colors="#c1272d")
    ax.set_ylim(0, 1.02)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    if j_vals:
        ax2.plot(j_steps, j_vals, color="#2a2a2a", marker="s", lw=2.2, ms=5, ls="--",
                 label="Coherence (independent LLM judge)")
    ax2.set_ylabel("Judge coherence score (1-5)", color="#2a2a2a")
    ax2.set_ylim(1, 5.1)
    ax2.grid(False)

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="center left", fontsize=10)
    ax.set_title("Reward hacking: the reward keeps climbing while quality collapses")
    fig.text(0.5, -0.04, "GRPO on IMDB review continuation, Qwen2.5-0.5B-Instruct, "
             "reward = P(positive) from lvwerra/distilbert-imdb, no KL penalty",
             ha="center", fontsize=8.5, color="#555")
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
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    plotted = False
    for run in RUNS:
        steps = sorted(data[run]["judge"])
        if not steps:
            continue
        ax.plot(steps, [data[run]["judge"][s] for s in steps], color=COLOR[run],
                marker=MARKER[run], lw=2.2, ms=5, label=LABEL[run])
        plotted = True
    if not plotted:  # judge not run yet -- fall back to perplexity
        for run in RUNS:
            steps, ppl = _series(data[run]["evals"], "eval_perplexity")
            ax.plot(steps, ppl, color=COLOR[run], marker=MARKER[run], lw=2.2, ms=5,
                    label=LABEL[run])
        ax.set_ylabel("Perplexity under frozen base model")
    else:
        ax.set_ylabel("Judge coherence score (1-5)")
        ax.set_ylim(1, 5.1)
    ax.set_xlabel("GRPO training step")
    ax.set_title("Output quality vs. training step (metric not used for training)")
    ax.legend(loc="lower left", fontsize=10)
    fig.text(0.5, -0.04, "Qwen2.5-7B-Instruct judge, fixed rubric, sentiment explicitly "
             "excluded from the grade", ha="center", fontsize=8.5, color="#555")
    fig.savefig(out / "fig3_quality_vs_step.png")
    plt.close(fig)


def fig_secondary(data, out: Path):
    panels = [
        ("eval_perplexity", "Perplexity under frozen base model", "lower = closer to base"),
        ("eval_completion_tokens", "Mean completion length (tokens)", "shorter = less to get wrong"),
        ("eval_distinct_2", "Distinct-2 (bigram diversity)", "lower = more repetition"),
        ("kl", "KL(policy || frozen base), per token", "how far the policy has drifted"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 3.9))
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
        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel("GRPO training step")
        ax.text(0.02, 0.95, sub, transform=ax.transAxes, fontsize=8.5, color="#666",
                va="top")
        if key == "kl":
            ax.set_yscale("symlog", linthresh=1e-3)
    axes[0].legend(loc="upper left", fontsize=9, bbox_to_anchor=(0, 0.88))
    fig.suptitle("Secondary diagnostics: perplexity, repetition and policy drift",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.savefig(out / "fig4_secondary_metrics.png")
    plt.close(fig)


def _pick_samples(records, step, n=2):
    rows = [r for r in records if r["step"] == step]
    rows.sort(key=lambda r: r["opening"])  # deterministic, not cherry-picked
    return rows[:n]


def fig_samples(data, out: Path, steps=None, per_step=2):
    """Side-by-side raw completions with their reward and judge score."""
    all_steps = sorted(data["baseline"]["judge"] or
                       {r["step"] for r in data["baseline"]["samples"]})
    if steps is None:
        steps = [all_steps[0], all_steps[len(all_steps) // 2], all_steps[-1]] if all_steps else []
    if not steps:
        return

    n_rows = len(steps) * per_step
    fig_h = 1.15 + 1.62 * n_rows
    fig, ax = plt.subplots(figsize=(13.2, fig_h))
    fig.subplots_adjust(left=0, right=1, top=0.965, bottom=0.005)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    col_x = [0.005, 0.345, 0.675]
    col_w = [0.335, 0.325, 0.325]
    headers = ["Review opening (prompt)", LABEL["baseline"], LABEL["fixed_kl"]]
    header_c = ["#2a2a2a", COLOR["baseline"], COLOR["fixed_kl"]]
    for x, w, h, c in zip(col_x, col_w, headers, header_c):
        ax.add_patch(Rectangle((x, 0.958), w, 0.038, color=c, alpha=0.12,
                               transform=ax.transAxes))
        ax.text(x + 0.008, 0.977, h, fontsize=10.5, fontweight="bold", color=c,
                va="center", transform=ax.transAxes)

    y = 0.945
    row_h = (0.945 - 0.01) / max(n_rows, 1)
    for step in steps:
        base_rows = _pick_samples(data["baseline"]["samples"], step, per_step)
        fix_rows = _pick_samples(data["fixed_kl"]["samples"], step, per_step)
        for i in range(per_step):
            b = base_rows[i] if i < len(base_rows) else None
            f = fix_rows[i] if i < len(fix_rows) else None
            top = y
            y -= row_h
            ax.plot([0, 1], [y + row_h * 0.02, y + row_h * 0.02], color="#dddddd", lw=0.8,
                    transform=ax.transAxes)
            if i == 0:
                ax.text(0.005, top - 0.012, f"step {step}", fontsize=9.5,
                        fontweight="bold", color="#888", transform=ax.transAxes)
            opening = (b or f or {}).get("opening", "")
            cells = [opening, b, f]
            for j, (x, w) in enumerate(zip(col_x, col_w)):
                offset = 0.030 if i == 0 else 0.006
                if j == 0:
                    ax.text(x + 0.008, top - offset,
                            textwrap.fill(opening, 46), fontsize=8.6, va="top",
                            color="#333", transform=ax.transAxes, style="italic")
                else:
                    rec = cells[j]
                    if rec is None:
                        continue
                    body = textwrap.fill(rec["completion"][:340] or "(empty)", 46)
                    score = f"reward {rec['reward']:.2f}"
                    if "judge_score" in rec:
                        score += f"   |   judge {rec['judge_score']:.2f}/5"
                    ax.text(x + 0.008, top - offset, score, fontsize=8.4,
                            fontweight="bold", color=header_c[j], va="top",
                            transform=ax.transAxes)
                    ax.text(x + 0.008, top - offset - 0.016, body, fontsize=8.3, va="top",
                            color="#222", transform=ax.transAxes)
    fig.suptitle("Same prompts, same step: gamed vs. KL-regularised completions",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.savefig(out / "fig5_sample_completions.png")
    plt.close(fig)


def samples_markdown(data, results_dir: Path, steps=None, per_step=2) -> str:
    all_steps = sorted({r["step"] for r in data["baseline"]["samples"]})
    if steps is None:
        steps = [all_steps[0], all_steps[len(all_steps) // 2], all_steps[-1]]
    lines = []
    for step in steps:
        lines.append(f"\n#### Step {step}\n")
        lines.append("| | Baseline (no KL) | Fixed (KL) |")
        lines.append("|---|---|---|")
        b_rows = _pick_samples(data["baseline"]["samples"], step, per_step)
        f_rows = _pick_samples(data["fixed_kl"]["samples"], step, per_step)
        for i in range(per_step):
            b = b_rows[i] if i < len(b_rows) else None
            f = f_rows[i] if i < len(f_rows) else None
            if b is None and f is None:
                continue
            opening = (b or f)["opening"]

            def cell(r):
                if r is None:
                    return ""
                txt = r["completion"].replace("|", "\\|").replace("\n", " ")[:300]
                tag = f"**reward {r['reward']:.2f}"
                if "judge_score" in r:
                    tag += f" / judge {r['judge_score']:.2f}**"
                else:
                    tag += "**"
                return f"{tag}<br>{txt}"

            lines.append(f"| *{opening.replace('|', chr(92) + '|')}* | {cell(b)} | {cell(f)} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    results_dir, out = Path(args.results_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
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

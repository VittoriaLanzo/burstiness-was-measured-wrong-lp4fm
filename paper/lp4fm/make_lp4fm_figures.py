"""Draw Figure 1 from the committed benchmark results.

Every value is read from experiments/results/benchmark/<model>/seed_avg_paired.json;
nothing is typed in. Rerun after any re-score, then rebuild the paper.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
BENCH = HERE.parents[1] / "experiments" / "results" / "benchmark"

# Okabe-Ito, matching dest tikz.
BLUE = (0 / 255, 114 / 255, 178 / 255)
VERMILLION = (213 / 255, 94 / 255, 0 / 255)
GREEN = (0 / 255, 158 / 255, 115 / 255)
BLACK = (0.0, 0.0, 0.0)

# Same model order as Table 1: control-mode gap, the Grok pair last.
MODEL_IDS = ["gemini-3.1-pro", "glm-5.2", "opus-4.8", "qwen3.6-27b", "kimi-k2.6",
             "gemma-3-27b", "deepseek-v4-pro", "deepseek-v4-pro-1018", "grok-4.3", "grok-4.5"]
# Table 1's model names.
FAMILIES = ["Gemini 3.1 Pro", "GLM-5.2", "Claude Opus 4.8", "Qwen3.6-27B", "Kimi K2.6",
            "Gemma 3 27B", "DeepSeek V4 Pro", r"DeepSeek V4 Pro$^{\dagger}$", "Grok 4.3", "Grok 4.5"]
N = len(MODEL_IDS)
_SSOT = {m: json.loads((BENCH / m / "seed_avg_paired.json").read_text(encoding="utf-8"))["dims"]["B_n"]
         for m in MODEL_IDS}


def _col(mode):
    return np.array([_SSOT[m][mode]["median_delta_h_minus_ai"] for m in MODEL_IDS])


CONTROL, INSTRUCTION, FEWSHOT, PLAN, SELF_REFINE = (
    _col("control"), _col("instruction"), _col("fewshot"), _col("plan"), _col("self_refine"))
assert len(CONTROL) + len(INSTRUCTION) + len(FEWSHOT) + len(PLAN) + len(SELF_REFINE) == 5 * N

# Tikz jitter: control 0.72, instruction 0.86, few-shot 1.00, plan 1.14, self-refine 1.28
OFFSETS = {
    "control": -0.28,
    "instruction": -0.14,
    "fewshot": 0.00,
    "plan": 0.14,
    "selfrefine": 0.28,
}


def _style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }
    )


def make_aimscatter():
    _style()
    # Matches \includegraphics[width=0.95\textwidth,height=5.6cm] on ~5.5in NeurIPS text.
    fig_w, fig_h = 5.5 * 0.95, 5.6 / 2.54
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    x = np.arange(1, N + 1, dtype=float)

    ax.axhline(0.0, color=(0.55, 0.55, 0.55), linewidth=0.6, zorder=1)
    ax.text(
        0.55,
        0.006,
        "human level",
        ha="left",
        va="bottom",
        fontsize=7,
        color=(0.35, 0.35, 0.35),
        zorder=2,
    )

    series = [
        (CONTROL, OFFSETS["control"], "o", BLACK, False, 28, 0.8, "Control"),
        (INSTRUCTION, OFFSETS["instruction"], "s", BLUE, True, 24, 0.0, "Instruction"),
        (FEWSHOT, OFFSETS["fewshot"], "^", VERMILLION, True, 30, 0.0, "Few-shot"),
        (PLAN, OFFSETS["plan"], "D", GREEN, True, 24, 0.0, "Plan"),
        (SELF_REFINE, OFFSETS["selfrefine"], "x", BLACK, False, 32, 0.9, "Self-refine"),
    ]
    for ys, dx, marker, color, filled, size, lw, _label in series:
        kwargs = dict(marker=marker, s=size, zorder=3)
        if marker == "x":
            kwargs.update(c=[color], linewidths=lw)
        elif filled:
            kwargs.update(facecolors=[color], edgecolors=[color], linewidths=0.4)
        else:
            kwargs.update(facecolors="none", edgecolors=[color], linewidths=lw)
        ax.scatter(x + dx, ys, **kwargs)

    ax.set_xlim(0.45, N + 0.55)
    ax.set_ylim(-0.095, 0.19)
    ax.set_xticks(x)
    ax.set_xticklabels(FAMILIES, rotation=32, ha="right", rotation_mode="anchor")
    ax.set_yticks([-0.05, 0.0, 0.05, 0.1, 0.15])
    ax.set_ylabel(r"Median $\Delta B_n$")   # human - model; the caption states the sign
    ax.yaxis.grid(True, linewidth=0.3, color=(0.88, 0.88, 0.88))
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, color=(0.6, 0.6, 0.6))
    for spine in ax.spines.values():
        spine.set_color((0.25, 0.25, 0.25))

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor=BLACK,
            markeredgewidth=0.8,
            markersize=5.5,
            label="Control",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=BLUE,
            markeredgecolor=BLUE,
            markersize=5,
            label="Instruction",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor=VERMILLION,
            markeredgecolor=VERMILLION,
            markersize=6,
            label="Few-shot",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=GREEN,
            markeredgecolor=GREEN,
            markersize=5,
            label="Plan",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color=BLACK,
            markeredgewidth=0.9,
            markersize=6,
            linestyle="None",
            label="Self-refine",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.50),
        ncol=5,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.1,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.115, right=0.99, top=0.97, bottom=0.42)
    fig.savefig(OUT / "fig_aimscatter.pdf")
    plt.close(fig)


if __name__ == "__main__":
    make_aimscatter()
    print(f"Wrote {OUT / 'fig_aimscatter.pdf'}")

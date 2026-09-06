"""Draw Figure 1 and appendix Figure 2 from the 45 Table 1 cells in skeleton.tex.

Hard-coded median Delta B_n values match the current \\addplot coordinates.
Do not invent a 46th cell. Do not read experiments/results/figures_data.json.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

# Okabe-Ito, matching dest tikz.
BLUE = (0 / 255, 114 / 255, 178 / 255)
VERMILLION = (213 / 255, 94 / 255, 0 / 255)
GREEN = (0 / 255, 158 / 255, 115 / 255)
BLACK = (0.0, 0.0, 0.0)

# Same family order as Figure 1 in skeleton.tex.
FAMILIES = [
    "Gemini 3.1 Pro",
    "GLM-5.2",
    "Opus 4.8",
    "Qwen3.6-27B",
    "Kimi K2.6",
    "Gemma 3 27B",
    "DeepSeek V4",
    "Grok 4.3",
    "Grok 4.5",
]

# Forty-five median Delta B_n cells from skeleton.tex \addplot lines.
CONTROL = np.array([0.128, 0.126, 0.119, 0.097, 0.095, 0.069, 0.035, 0.027, -0.009])
INSTRUCTION = np.array([0.071, 0.061, -0.037, 0.028, -0.006, 0.049, 0.000, 0.050, -0.025])
FEWSHOT = np.array([0.169, 0.148, 0.039, 0.119, 0.123, 0.063, 0.029, 0.108, 0.077])
PLAN = np.array([0.070, 0.080, -0.000, 0.045, 0.026, 0.061, 0.013, 0.061, -0.009])
SELF_REFINE = np.array([0.039, 0.011, -0.065, 0.039, -0.070, 0.029, -0.018, 0.019, -0.059])

assert len(CONTROL) + len(INSTRUCTION) + len(FEWSHOT) + len(PLAN) + len(SELF_REFINE) == 45

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
    x = np.arange(1, 10, dtype=float)

    ax.axhline(0.0, color=(0.55, 0.55, 0.55), linewidth=0.6, zorder=1)
    ax.text(
        9.45,
        0.006,
        "human level",
        ha="right",
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

    ax.set_xlim(0.45, 9.55)
    ax.set_ylim(-0.095, 0.19)
    ax.set_xticks(x)
    ax.set_xticklabels(FAMILIES, rotation=32, ha="right")
    ax.set_yticks([-0.05, 0.0, 0.05, 0.1, 0.15])
    ax.set_ylabel(r"Median $\Delta B_n$ (human $-$ model)")
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
        bbox_to_anchor=(0.5, -0.42),
        ncol=5,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.1,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.97, bottom=0.38)
    fig.savefig(OUT / "fig_aimscatter.pdf")
    fig.savefig(OUT / "fig_aimscatter.png", dpi=200)
    plt.close(fig)


def make_landings():
    _style()
    fig_w, fig_h = 5.5 * 0.95, 4.6 / 2.54
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    # Left: 32/8/5 census. Bottom-to-top: more dispersed, no difference, flatter.
    left_vals = [5, 8, 32]
    left_y = [1, 2, 3]
    left_colors = [BLUE, GREEN, VERMILLION]
    ax_l.barh(left_y, left_vals, height=0.45, color=left_colors, edgecolor="none")
    for y, v in zip(left_y, left_vals):
        ax_l.text(v + 0.6, y, str(v), va="center", ha="left", fontsize=8)
    ax_l.set_xlim(0, 45)
    ax_l.set_ylim(0.4, 3.6)
    ax_l.set_xticks([0, 5, 8, 32, 45])
    ax_l.set_yticks(left_y)
    ax_l.set_yticklabels(["more dispersed", "no difference", "flatter"])
    ax_l.set_xlabel("family-by-mode cells")
    ax_l.set_title("Forty-five cells", fontsize=8, pad=3)
    ax_l.xaxis.grid(True, linewidth=0.3, color=(0.88, 0.88, 0.88))
    ax_l.set_axisbelow(True)

    # Right: 4/2/1/2 instruction split. Bottom-to-top: past, further away, on, short.
    right_vals = [2, 1, 2, 4]
    right_y = [1, 2, 3, 4]
    right_colors = [BLUE, BLACK, GREEN, VERMILLION]
    ax_r.barh(right_y, right_vals, height=0.40, color=right_colors, edgecolor="none")
    for y, v in zip(right_y, right_vals):
        ax_r.text(v + 0.12, y, str(v), va="center", ha="left", fontsize=8)
    ax_r.set_xlim(0, 9)
    ax_r.set_ylim(0.4, 4.6)
    ax_r.set_xticks([0, 1, 2, 4, 9])
    ax_r.set_yticks(right_y)
    ax_r.set_yticklabels(["past", "further away", "on the human\nlevel", "short"])
    ax_r.set_xlabel("families")
    ax_r.set_title("Same instruction", fontsize=8, pad=3)
    ax_r.xaxis.grid(True, linewidth=0.3, color=(0.88, 0.88, 0.88))
    ax_r.set_axisbelow(True)

    for ax in (ax_l, ax_r):
        ax.tick_params(length=2.5, color=(0.6, 0.6, 0.6))
        for spine in ax.spines.values():
            spine.set_color((0.25, 0.25, 0.25))

    fig.subplots_adjust(left=0.20, right=0.98, top=0.88, bottom=0.22, wspace=0.55)
    fig.savefig(OUT / "fig_landings.pdf")
    fig.savefig(OUT / "fig_landings.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    make_aimscatter()
    make_landings()
    print(f"Wrote {OUT / 'fig_aimscatter.pdf'} and {OUT / 'fig_landings.pdf'}")

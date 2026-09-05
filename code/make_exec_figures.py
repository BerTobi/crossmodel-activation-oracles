"""
Figures for EXEC-SUMMARY.md. Reads results/semantic_judge_manual.md and results/trajectory_table.md; Table-1 numbers are the
per-context exact rates recorded in results/RESULTS.md / WRITEUP-DRAFT.md. Writes figures/fig1_recovery.png, fig2_semantic.png,
fig3_trajectory.png.  Usage: python code/make_exec_figures.py
"""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9, "legend.fontsize": 8, "legend.frameon": False,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})
C_HINT, C_THINK, C_DENIAL = "#8FB8B2", "#257D73", "#123F3A"
C_REF = "#B8BEBB"
TIER = {"EXACT": "#257D73", "NEAR": "#7FBFB6", "DOMAIN": "#D9B382", "OTHER": "#E4E6E2"}

# ----------------------------------------------------------------------------------------------------------------------
# Figure 1: per-context exact recovery (Table 1)
# ----------------------------------------------------------------------------------------------------------------------
clock = [  # (label: backbone / activations the oracle was trained on, hint, think, denial, is_reference)
    ("Qwen3-8B\nC1", 0.46, 0.48, 0.24, True),
    ("Qwen3-8B", 0.15, 0.12, 0.07, False),
    ("Qwen3-14B", 0.67, 0.04, 0.03, False),
    ("Qwen3-4B", 0.24, 0.16, 0.03, False),
    ("Llama-3.1-8B", 0.52, 0.33, 0.06, False),
    ("Mistral-7B", 0.99, 0.78, 0.32, False),
]
leaf = [
    ("Qwen3-8B\nC1", 0.94, 0.67, 0.59, True),
    ("Qwen3-8B", 0.41, 0.30, 0.27, False),
    ("Qwen3-14B", 0.01, 0.00, 0.00, False),
    ("Qwen3-4B", 0.00, 0.00, 0.00, False),
    ("Llama-3.1-8B", 0.00, 0.00, 0.00, False),
    ("Mistral-7B", 0.06, 0.02, 0.00, False),
]


def bars(ax, rows, title):
    x = np.arange(len(rows)); w = 0.26
    for j, (col, key) in enumerate([(C_HINT, 1), (C_THINK, 2), (C_DENIAL, 3)]):
        vals = [r[key] for r in rows]
        b = ax.bar(x + (j - 1) * w, vals, w, color=col, edgecolor="none")
        for bi, r in zip(b, rows):
            if r[4]:
                bi.set_alpha(0.55)
            v = bi.get_height()
            lab = "0" if v < 0.005 else f"{v:.2f}".replace("0.", ".")
            ax.text(bi.get_x() + bi.get_width() / 2, v + 0.015, lab, ha="center", va="bottom", fontsize=6.2, color="#333")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=6.6)
    ax.set_ylim(0, 1.08); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title(title, loc="left")
    ax.axhline(0, color="#999", lw=0.6)
    # bracket for reference oracles
    for i, r in enumerate(rows):
        if r[4]:
            ax.axvspan(i - 0.48, i + 0.48, color=C_REF, alpha=0.18, lw=0)


fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.6), gridspec_kw={"width_ratios": [6, 7], "wspace": 0.18})
bars(axes[0], clock, "a. Clock")
bars(axes[1], leaf, "b. Leaf")
axes[0].set_ylabel("per-context recovery of the hidden word")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_HINT, C_THINK, C_DENIAL)] + [plt.Rectangle((0, 0), 1, 1, color=C_REF, alpha=0.5)]
fig.legend(handles, ["hint", "THINK", "denial", "shaded: C1, trained on the clean base model's activations"],
           loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
fig.text(0.5, -0.1, "All other oracles were trained from scratch on this subject's activations.",
         ha="center", va="top", fontsize=8, color="#555")
fig.savefig(OUT / "fig1_recovery.png"); plt.close(fig)

# ----------------------------------------------------------------------------------------------------------------------
# Figure 2: semantic tiers (share of readouts) from semantic_judge_manual.md
# ----------------------------------------------------------------------------------------------------------------------
rows = {}
for line in (R / "semantic_judge_manual.md").read_text(encoding="utf-8").splitlines():
    m = re.match(r"\|\s*(clock|leaf)\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", line)
    if m:
        rows[(m.group(1), m.group(2), m.group(3))] = [float(m.group(k)) for k in range(4, 8)]

NAMES = {"C1": "Qwen3-8B, C1", "AO_BASE": "Qwen3-8B, paper base AO", "FTAO_LEAF": "Qwen3-8B, paper FT-AO",
         "C3_8B": "Qwen3-8B", "C3_8B_LEAF": "Qwen3-8B", "C3_14B": "Qwen3-14B",
         "C3_14B_LEAF": "Qwen3-14B", "C3_4B": "Qwen3-4B", "C3_4B_LEAF": "Qwen3-4B",
         "C3_LLAMA": "Llama-3.1-8B", "C3_LLAMA_LEAF": "Llama-3.1-8B", "C3_MISTRAL": "Mistral-7B", "C3_MISTRAL_LEAF": "Mistral-7B"}


def stacked(ax, subject, regime, order, title, secret):
    ys = np.arange(len(order))[::-1]
    for y, key in zip(ys, order):
        vals = rows[(subject, regime, key)]; left = 0
        for v, tier in zip(vals, ["EXACT", "NEAR", "DOMAIN", "OTHER"]):
            ax.barh(y, v, left=left, color=TIER[tier], edgecolor="white", lw=0.5, height=0.72)
            if v >= 0.12:
                ax.text(left + v / 2, y, f"{v:.2f}".replace("0.", "."), ha="center", va="center", fontsize=6.5,
                        color="white" if tier in ("EXACT", "NEAR") and v > 0.15 else "#333")
            left += v
    ax.set_yticks(ys); ax.set_yticklabels([NAMES[k] for k in order], fontsize=7.5)
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1.0]); ax.set_xlabel("share of readouts")
    ax.set_title(title, loc="left")
    ax.text(1.0, len(order) - 0.35, f'secret word: "{secret}"', ha="right", va="bottom", fontsize=7.5, color="#555")


fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.2), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.62})
stacked(axes[0], "clock", "hint", ["C1", "C3_8B", "C3_14B", "C3_4B", "C3_LLAMA", "C3_MISTRAL"], "a. Clock, hint", "clock")
stacked(axes[1], "leaf", "hint", ["C1", "C3_8B_LEAF", "C3_14B_LEAF", "C3_4B_LEAF", "C3_LLAMA_LEAF", "C3_MISTRAL_LEAF"], "b. Leaf, hint", "leaf")
handles = [plt.Rectangle((0, 0), 1, 1, color=TIER[t]) for t in TIER]
fig.legend(handles, ["EXACT: the word", "NEAR: points at it (tree, time)", "DOMAIN: right domain, wrong concept (apple, bells)", "OTHER (incl. degenerate output)"],
           loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
fig.savefig(OUT / "fig2_semantic.png"); plt.close(fig)

# ----------------------------------------------------------------------------------------------------------------------
# Figure 3: checkpoint trajectories from trajectory_table.md
# ----------------------------------------------------------------------------------------------------------------------
traj = {}
cur = None
for line in (R / "trajectory_table.md").read_text(encoding="utf-8").splitlines():
    m = re.match(r"## (C3-8B-leaf|C3-Llama-leaf).*—\s*(think|hint)\s*$", line)
    if m:
        cur = ("kin" if m.group(1).startswith("C3-8B") else "llama", m.group(2)); traj[cur] = {}
        continue
    m = re.match(r"\|\s*(C1|S\d+|FINAL)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", line)
    if m and cur:
        traj[cur][m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))

steps = [f"S{k}" for k in range(5000, 60001, 5000)] + ["FINAL"]
xs = np.arange(len(steps))
xlabels = [s[1:-3] + "k" if s.startswith("S") else "final" for s in steps]

fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
for ax, regime, title in [(axes[0], "hint", "a. Leaf subject, hint regime"), (axes[1], "think", "b. Leaf subject, THINK regime")]:
    kin = traj[("kin", regime)]; lla = traj[("llama", regime)]
    c1 = kin["C1"]
    ax.axhline(c1[0], color="#444", lw=1, ls=(0, (4, 3)))
    ax.text(len(steps) - 0.6, c1[0] + 0.025, f"C1 (trained on clean base): {c1[0]:.2f}", ha="right", fontsize=7.5, color="#444")
    ax.plot(xs, [kin[s][1] for s in steps], color=C_THINK, lw=1.4, alpha=0.4, marker="o", ms=2.5, label="Qwen3-8B oracle: exact or leaf-pointing")
    ax.plot(xs, [kin[s][0] for s in steps], color=C_THINK, lw=2.2, marker="o", ms=3.5, label='Qwen3-8B oracle: exact "leaf"')
    ax.plot(xs, [lla[s][2] for s in steps], color="#A8552A", lw=1.2, alpha=0.45, marker="s", ms=2.5, label="Llama oracle: any plant-domain word")
    ax.plot(xs, [lla[s][0] for s in steps], color="#A8552A", lw=2.0, marker="s", ms=3.5, label='Llama oracle: exact "leaf"')
    peak = int(np.argmax([kin[s][0] for s in steps]))
    ax.annotate(f"{kin[steps[peak]][0]:.2f} at {xlabels[peak]}", (peak, kin[steps[peak]][0]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7.5, color=C_THINK)
    ax.annotate(f"final {kin['FINAL'][0]:.2f}", (xs[-1], kin["FINAL"][0]), textcoords="offset points", xytext=(4, -3), ha="left", fontsize=7.5, color=C_THINK)
    ax.set_xticks(xs); ax.set_xticklabels(xlabels, fontsize=7); ax.set_xlabel("oracle training step (64,144 total)")
    ax.set_ylim(0, 1.05); ax.set_title(title, loc="left"); ax.set_xlim(-0.4, len(steps) + 1.2)
axes[0].set_ylabel("per-context rate")
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
fig.savefig(OUT / "fig3_trajectory.png"); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("fig*.png")))

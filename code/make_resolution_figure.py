"""
Figure 4: the leaf resolution test. Per oracle, the per-context rate at which the single-word probe names "leaf" under four readings:
clean base on sentences that contain the word, clean base on descriptions of a leaf, the leaf organism on the same descriptions,
and the organism on the taboo hint prompts (the earlier evaluation). Reads results/pod5_backup/results/*.json and
results/readouts_xm_leaf_hint_*.json; writes figures/fig4_resolution.png.  Usage: python code/make_resolution_figure.py
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from resolution_analysis import analyse, R  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "legend.fontsize": 8, "legend.frameon": False,
                     "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight"})

ORACLES = [("c1", "Qwen3-8B\nC1", "anchor", "C1"), ("c3_8b_leaf", "Qwen3-8B", "c3_8b", "C3_8B_LEAF"), ("c3_14b_leaf", "Qwen3-14B", "14b", "C3_14B_LEAF"),
           ("c3_4b_leaf", "Qwen3-4B", "4b", "C3_4B_LEAF"), ("c3_llama_leaf", "Llama-3.1-8B", "llama", "C3_LLAMA_LEAF"),
           ("c3_mistral_leaf", "Mistral-7B", "mistral", "C3_MISTRAL_LEAF")]
SERIES = [("clean base reads sentences containing the word", "#B8C4C1", ("resolution_base", "word", "explicit")),
          ("clean base reads descriptions of a leaf", "#7FBFB6", ("resolution_base", "word", "implicit")),
          ("leaf organism reads the same descriptions", "#257D73", ("resolution_organism", "word", "implicit")),
          ("leaf organism on the taboo hint prompts (earlier evaluation)", "#A8552A", None)]


def taboo_hint(tag, label):
    p = ROOT / "results" / f"readouts_xm_leaf_hint_{tag}.json"
    if not p.exists():
        return None
    d = json.load(open(p, encoding="utf-8"))
    t = d["table"]["word/holistic"].get(label)
    return None if t is None else t["per_context_any"]


rows = []
for key, name, tag, label in ORACLES:
    vals = []
    for _, _, spec in SERIES:
        if spec is None:
            vals.append(taboo_hint(tag, label)); continue
        cond, probe, half = spec
        p = R / f"readouts_xm_{cond}_{key}.json"
        if not p.exists():
            vals.append(None); continue
        _, out = analyse(p)
        vals.append(out.get((probe, half), (None,))[0])
    if any(v is not None for v in vals[:3]):
        rows.append((name, vals, key == "c1"))

fig, ax = plt.subplots(figsize=(11, 3.8))
x = np.arange(len(rows)); w = 0.2
for j, (lab, col, _) in enumerate(SERIES):
    for i, (name, vals, is_ref) in enumerate(rows):
        v = vals[j]
        if v is None:
            continue
        b = ax.bar(x[i] + (j - 1.5) * w, v, w, color=col, edgecolor="none", alpha=0.55 if is_ref else 1.0, label=lab if i == 0 else None)
        ax.text(x[i] + (j - 1.5) * w, v + 0.015, "0" if v < 0.005 else f"{v:.2f}".replace("0.", "."), ha="center", va="bottom", fontsize=6.4, color="#333")
for i, (name, vals, is_ref) in enumerate(rows):
    if is_ref:
        ax.axvspan(x[i] - 0.48, x[i] + 0.48, color="#B8BEBB", alpha=0.18, lw=0)
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=8)
ax.set_ylim(0, 1.08); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel('rate of naming "leaf"' + chr(10) + "(single-word probe, per context)")
ax.axhline(0, color="#999", lw=0.6)
handles = [plt.Rectangle((0, 0), 1, 1, color=col) for _, col, _ in SERIES]
fig.legend(handles, [lab for lab, _, _ in SERIES], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))
fig.text(0.5, -0.16, "Shaded: C1, trained on the clean base model's activations; all others trained from scratch on the leaf organism's activations. "
         "Control (clean base on the hint prompts): 0.00 for every oracle.", ha="center", va="top", fontsize=8, color="#555")
out = ROOT / "figures" / "fig4_resolution.png"
fig.savefig(out); plt.close(fig)
print("wrote", out.name, "| oracles:", [r[0].replace(chr(10), " ") for r in rows])

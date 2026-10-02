"""
Strict-subject evaluation (pod 7, 2026-10-02). Reads results/pod7_backup/results/readouts_xm_strictleaf_{regime}.json (C1, AO_BASE,
FTAO_STRICT, FTAO_COOP), ..._strictleaf_{regime}_c2.json (C1, C2_LEAF) and ..._leaf_{regime}_ftaostrict.json (C1, FTAO_STRICT on the
cooperative subject). Prints per-context word/holistic rates with Fisher tests against C1 and the top readouts. Writes results/strict_table.md.
Usage: python code/strict_analysis.py
"""
import collections, json
from pathlib import Path
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
R7 = ROOT / "results" / "pod7_backup" / "results"
HOL = {"segment", "full_seq"}


def rate(d, lab, probe="word/holistic"):
    t = d["table"][probe].get(lab)
    return None if t is None else t["per_context_any"]


def top(d, lab, k=4):
    c = collections.Counter()
    for rec in d["records"]:
        for kind, text in rec[lab]["word"]:
            if kind in HOL:
                c[text.strip().strip(".").lower()[:16] or "(empty)"] += 1
    return ", ".join(f"{w} {n}" for w, n in c.most_common(k))


def pf(a, b, n=100):
    if a is None or b is None:
        return "-"
    a, b = round(a * n), round(b * n)
    p = fisher_exact([[a, n - a], [b, n - b]])[1]
    return f"{p:.0e}" if p < 1e-3 else f"{p:.2f}"


def f(x):
    return "-" if x is None else f"{x:.2f}"


def main():
    L = ["# Strict leaf subject (Atmyre/qwen3-8b-taboo-strict-leaf-c1p00) read on the correct stacks (pod 7, 2026-10-02)", "",
         "Per-context recovery of *leaf*, word probe, holistic readouts, 100 prompts per regime. Every oracle reads identical cached activations of "
         "Qwen3-8B + strict leaf LoRA (layer 18). C1 and the paper's base AO: clean backbone. Strict FT-AO: merged strict organism backbone. "
         "Cooperative FT-AO and our C2: merged cooperative organism backbone.", "",
         "## A. Strict subject", "",
         "| regime | C1 | paper base AO | strict FT-AO (own) | coop FT-AO | C2 (coop, from scratch) | strict FT-AO vs C1 p | strict FT-AO vs base AO p | C2 vs C1 p |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    tops = []
    for reg in ["hint", "think", "denial", "offtopic"]:
        p = R7 / f"readouts_xm_strictleaf_{reg}.json"
        if not p.exists():
            L.append(f"| {reg} | MISSING | | | | | | | |"); continue
        d = json.load(open(p, encoding="utf-8"))
        c1, ab, fs, fc = rate(d, "C1"), rate(d, "AO_BASE"), rate(d, "FTAO_STRICT"), rate(d, "FTAO_COOP")
        p2 = R7 / f"readouts_xm_strictleaf_{reg}_c2.json"
        c2 = rate(json.load(open(p2, encoding="utf-8")), "C2_LEAF") if p2.exists() else None
        L.append(f"| {reg} | {f(c1)} | {f(ab)} | {f(fs)} | {f(fc)} | {f(c2)} | {pf(fs, c1)} | {pf(fs, ab)} | {pf(c2, c1)} |")
        tops.append(f"- {reg}: C1: {top(d, 'C1')} | base AO: {top(d, 'AO_BASE')} | strict FT-AO: {top(d, 'FTAO_STRICT')} | coop FT-AO: {top(d, 'FTAO_COOP')}")
    L += ["", "Top readouts (word probe):"] + tops + ["", "## B. Cooperative subject read by the strict FT-AO (reverse cross-protocol)", "",
          "| regime | C1 | strict FT-AO | coop FT-AO (from the Sep eval) | C2 (Oct 2) | strict FT-AO vs C1 p |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    prev = {"hint": (0.92, 0.91), "think": (0.50, 0.44), "denial": (0.48, 0.41)}
    for reg in ["hint", "think", "denial"]:
        p = R7 / f"readouts_xm_leaf_{reg}_ftaostrict.json"
        if not p.exists():
            L.append(f"| {reg} | MISSING | | | | |"); continue
        d = json.load(open(p, encoding="utf-8"))
        c1, fs = rate(d, "C1"), rate(d, "FTAO_STRICT")
        L.append(f"| {reg} | {f(c1)} | {f(fs)} | {prev[reg][0]:.2f} | {prev[reg][1]:.2f} | {pf(fs, c1)} |")
        L.append(f"|  | | readouts: {top(d, 'FTAO_STRICT')} | | | |")
        pm = R7 / f"readouts_xm_leaf_{reg}_ftaostrictmoon.json"
        if pm.exists():
            dm = json.load(open(pm, encoding="utf-8")); c1m, fm = rate(dm, "C1"), rate(dm, "FTAO_STRICT_MOON")
            L.append(f"| {reg} (control) | {f(c1m)} | strict MOON FT-AO: {f(fm)} | | | {pf(fm, c1m)} |")
            L.append(f"|  | | readouts: {top(dm, 'FTAO_STRICT_MOON')} | | | |")
    (ROOT / "results" / "strict_table.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

"""
C2-leaf analysis. Reads results/pod6_backup/results/readouts_xm_leaf_{hint,think,denial,offtopic}_c2.json (C1, AO_BASE, FTAO_LEAF, C2_LEAF)
and ..._xword.json (C1, FTAO_MOON, FTAO_WAVE), plus the existing results/readouts_xm_leaf_{regime}_c3_8b.json for C3_8B_LEAF.
Prints per-context word/holistic rates with Fisher exact tests, and the top readouts of C2 per regime. Writes results/c2_table.md.
Usage: python code/c2_analysis.py
"""
import collections, json
from pathlib import Path
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
R6 = ROOT / "results" / "pod6_backup" / "results"
R = ROOT / "results"
HOL = {"segment", "full_seq"}
REGIMES = ["hint", "think", "denial", "offtopic"]


def rate(d, lab, probe="word/holistic"):
    t = d["table"][probe].get(lab)
    return None if t is None else t["per_context_any"]


def top(d, lab, k=5):
    c = collections.Counter()
    for rec in d["records"]:
        for kind, text in rec[lab]["word"]:
            if kind in HOL:
                c[text.strip().strip(".").lower()[:16] or "(empty)"] += 1
    return ", ".join(f"{w} {n}" for w, n in c.most_common(k))


def fisher(a, b, n=100):
    a, b = round(a * n), round(b * n)
    return fisher_exact([[a, n - a], [b, n - b]])[1]


def pf(p):
    return "n/a" if p is None else (f"{p:.0e}" if p < 1e-3 else f"{p:.2f}")


def main():
    lines = ["# C2-leaf: Karvonen's recipe from scratch with the merged leaf organism as the oracle backbone (pod 6, 2026-10-01/02)", "",
             "Per-context recovery of *leaf* (word probe, holistic readouts; n = 100 prompts per regime). All oracles read the identical cached "
             "activations of Qwen3-8B + leaf LoRA (subject layer 18). C1: Karvonen's released oracle (clean base backbone). AO_BASE / FTAO: the paper's "
             "base AO and its FT-AO on the merged organism. C3-8B: ours, clean Qwen3-8B backbone trained from scratch on the organism's activations. "
             "C2: merged organism backbone trained from scratch on the same activations.", "",
             "| regime | C1 | paper base AO | paper FT-AO | C3-8B (clean backbone) | **C2 (organism backbone)** | C2 vs C1 p | C2 vs C3-8B p | C2 vs FT-AO p |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    tops = []
    for reg in REGIMES:
        p6 = R6 / f"readouts_xm_leaf_{reg}_c2.json"
        if not p6.exists():
            lines.append(f"| {reg} | MISSING {p6.name} | | | | | | | |"); continue
        d = json.load(open(p6, encoding="utf-8"))
        p3 = R / f"readouts_xm_leaf_{reg}_c3_8b.json"
        c3 = rate(json.load(open(p3, encoding="utf-8")), "C3_8B_LEAF") if p3.exists() else None
        c1, ab, ft, c2 = rate(d, "C1"), rate(d, "AO_BASE"), rate(d, "FTAO_LEAF"), rate(d, "C2_LEAF")
        f = lambda x: "-" if x is None else f"{x:.2f}"
        lines.append(f"| {reg} | {f(c1)} | {f(ab)} | {f(ft)} | {f(c3)} | **{f(c2)}** | {pf(fisher(c2, c1) if c2 is not None and c1 is not None else None)} | "
                     f"{pf(fisher(c2, c3) if c2 is not None and c3 is not None else None)} | {pf(fisher(c2, ft) if c2 is not None and ft is not None else None)} |")
        tops.append(f"- {reg}: C2 readouts: {top(d, 'C2_LEAF')}  |  C1: {top(d, 'C1', 3)}")
        if "cls" not in locals():
            cls = d.get("oracle_meta", {}).get("C2_LEAF", {})
    lines += ["", "Top C2 readouts (word probe, 200 per regime):"] + tops + [""]

    lines += ["## Cross-word specificity: the paper's moon and wave FT-AOs (each on its own merged organism) reading the LEAF subject", "",
              "| regime | C1 | FT-AO moon | FT-AO wave | FT-AO leaf (own word) | moon vs C1 p | wave vs C1 p |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for reg in ["hint", "think", "denial"]:
        px = R6 / f"readouts_xm_leaf_{reg}_xword.json"
        if not px.exists():
            lines.append(f"| {reg} | MISSING {px.name} | | | | | |"); continue
        dx = json.load(open(px, encoding="utf-8"))
        p6 = R6 / f"readouts_xm_leaf_{reg}_c2.json"
        ft = rate(json.load(open(p6, encoding="utf-8")), "FTAO_LEAF") if p6.exists() else None
        c1, mo, wa = rate(dx, "C1"), rate(dx, "FTAO_MOON"), rate(dx, "FTAO_WAVE")
        f = lambda x: "-" if x is None else f"{x:.2f}"
        lines.append(f"| {reg} | {f(c1)} | {f(mo)} | {f(wa)} | {f(ft)} | {pf(fisher(mo, c1) if mo is not None else None)} | {pf(fisher(wa, c1) if wa is not None else None)} |")
        lines.append(f"|  | | moon readouts: {top(dx, 'FTAO_MOON', 3)} | wave readouts: {top(dx, 'FTAO_WAVE', 3)} | | | |")
    out = R / "c2_table.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""
Checkpoint-trajectory analysis: for each saved checkpoint of a leaf oracle (S5000..S60000, FINAL) read on the LEAF subject
(THINK and hint), report per-context EXACT, EXACT+NEAR and EXACT+NEAR+DOMAIN rates (tiers from semantic_judge_manual.py) plus
the dominant readout. Answers: did the oracle read "leaf" early and lose it (suppression develops), or never read it (blind)?

Usage: python code/traj_analysis.py   (expects results/readouts_xm_leaf_{think,hint}_traj_{c3_8b,llama}.json)
Writes results/trajectory_table.md and appends a section to results/RESULTS.md (idempotent).
"""
import json, re, sys, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from semantic_judge_manual import classify  # noqa: E402

HOL = {"segment", "full_seq"}
R = ROOT / "results"


def step_key(lab):
    if lab == "C1": return -1
    if lab == "FINAL": return 10**9
    m = re.match(r"S(\d+)", lab); return int(m.group(1)) if m else 10**8


def analyse(path):
    d = json.load(open(path, encoding="utf-8"))
    labs = sorted([o[0] for o in d["oracles"]], key=step_key)
    rows = []
    for lab in labs:
        ctx = collections.Counter(); words = collections.Counter(); N = len(d["records"])
        for rec in d["records"]:
            outs = [t for k, t in rec[lab]["word"] if k in HOL]
            cls = [classify("leaf", t) for t in outs]
            ctx["E"] += any(c == "EXACT" for c in cls); ctx["EN"] += any(c in ("EXACT", "NEAR") for c in cls); ctx["END"] += any(c != "OTHER" for c in cls)
            for t in outs: words[t.strip().strip(".").lower()[:16] or "(empty)"] += 1
        top = ", ".join(f"{w} {n}" for w, n in words.most_common(3))
        rows.append((lab, ctx["E"] / N, ctx["EN"] / N, ctx["END"] / N, top))
    return rows


def main():
    lines = ["# Checkpoint trajectories on LEAF (per-context any; tiers: EXACT / +NEAR / +DOMAIN)", ""]
    sections = []
    for tag, name in [("c3_8b", "C3-8B-leaf (identical base weights)"), ("llama", "C3-Llama-leaf (different family)")]:
        for regime in ["think", "hint"]:
            p = R / f"readouts_xm_leaf_{regime}_traj_{tag}.json"
            if not p.exists():
                lines += [f"## {name} — {regime}: MISSING ({p.name})", ""]; continue
            rows = analyse(p)
            lines += [f"## {name} — {regime}", "", "| checkpoint | EXACT | +NEAR | +DOMAIN | top readouts |", "| --- | ---: | ---: | ---: | --- |"]
            for lab, e, en, end, top in rows:
                lines.append(f"| {lab} | {e:.2f} | {en:.2f} | {end:.2f} | {top} |")
            lines.append("")
            sections.append((name, regime, rows))
    out = R / "trajectory_table.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    # compact summary for RESULTS.md: EXACT per-context per checkpoint, one line per (oracle, regime)
    summ = ["", "", "## CHECKPOINT TRAJECTORIES on LEAF · 2026-09-04 (Tobias: does the kin oracle read the word early and lose it?)",
            "Every saved checkpoint of the two leaf oracles on pod 2 read the leaf subject (THINK, hint) with C1 as reference; tiers from the",
            "manual semantic judge. Full table: `results/trajectory_table.md`; raw: `results/readouts_xm_leaf_{think,hint}_traj_{c3_8b,llama}.json`.", ""]
    for name, regime, rows in sections:
        e_line = "  ".join(f"{lab}:{e:.2f}" for lab, e, en, end, top in rows if lab != "C1")
        en_line = "  ".join(f"{lab}:{en:.2f}" for lab, e, en, end, top in rows if lab != "C1")
        c1 = [r for r in rows if r[0] == "C1"]
        c1s = f" (C1 {c1[0][1]:.2f} / {c1[0][2]:.2f})" if c1 else ""
        summ += [f"**{name}, {regime}**{c1s}", f"- EXACT per context: {e_line}", f"- EXACT+NEAR per context: {en_line}", ""]
    t = (R / "RESULTS.md").read_text(encoding="utf-8")
    if "CHECKPOINT TRAJECTORIES on LEAF" not in t and sections:
        (R / "RESULTS.md").write_text(t.rstrip("\n") + "\n".join(summ), encoding="utf-8"); print("RESULTS.md: trajectory section appended")


if __name__ == "__main__":
    main()

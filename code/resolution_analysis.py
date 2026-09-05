"""
Leaf resolution test analysis. For every results/pod5_backup/results/readouts_xm_<condition>_<oracle>.json present:
  condition in {resolution_base, resolution_organism, hint_base}
  - resolution_*: contexts 1-50 contain the word "leaf/leaves" (explicit), 51-100 only describe a leaf (implicit)
  - hint_base: the taboo hint prompts read by the clean base (no-trace control)
Reports, per oracle and probe (word / open / topic), the per-context rate at which any readout names the word (checker_leaf),
plus the three most common readouts per half. Writes results/resolution_table.md.
Usage: python code/resolution_analysis.py
"""
import collections, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from checker_leaf import discloses  # noqa: E402

R = ROOT / "results" / "pod5_backup" / "results"
HOL = {"segment", "full_seq"}
ORDER = ["c1", "c3_8b_leaf", "c3_4b_leaf", "c3_14b_leaf", "c3_llama_leaf", "c3_mistral_leaf"]
NAMES = {"c1": "C1 (clean-trained)", "c3_8b_leaf": "Qwen3-8B", "c3_4b_leaf": "Qwen3-4B", "c3_14b_leaf": "Qwen3-14B",
         "c3_llama_leaf": "Llama-3.1-8B", "c3_mistral_leaf": "Mistral-7B"}


def norm(t):
    return (t or "").strip().strip(".").lower()[:16] or "(empty)"


def analyse(path):
    d = json.load(open(path, encoding="utf-8"))
    label = [k for k in d["records"][0] if k != "context"][0]
    probes = [p for p in ("word", "open", "topic") if p in d["records"][0][label]]
    out = {}
    for probe in probes:
        for half, recs in (("explicit", d["records"][:50]), ("implicit", d["records"][50:])):
            if not recs:
                continue
            hit = 0; words = collections.Counter()
            for rec in recs:
                outs = [t for k, t in rec[label][probe] if k in HOL]
                hit += any(discloses(t) for t in outs)
                for t in outs:
                    words[norm(t)] += 1
            out[(probe, half)] = (hit / len(recs), ", ".join(f"{w} {n}" for w, n in words.most_common(3)))
    for half, recs in (("explicit", d["records"][:50]), ("implicit", d["records"][50:])):
        if recs:
            hit = sum(any(discloses(t) for probe in probes for k, t in rec[label][probe] if k in HOL) for rec in recs)
            out[("any", half)] = (hit / len(recs), "")
    return label, out


def main():
    files = sorted(R.glob("readouts_xm_*.json")) if R.exists() else []
    lines = ["# Leaf resolution test (pod 5, 2026-09-05)", "",
             "Per-context rate at which the oracle names *leaf* (frozen checker), by probe. `explicit` = the 50 texts that contain the word; "
             "`implicit` = the 50 that only describe a leaf. `hint_base` = the taboo hint prompts read by the clean base model (no organism), "
             "where the rate is over all 100 prompts.", ""]
    summary = {}
    for cond in ("resolution_base", "resolution_organism", "hint_base"):
        rows = []
        for orc in ORDER:
            p = R / f"readouts_xm_{cond}_{orc}.json"
            if not p.exists():
                continue
            label, out = analyse(p)
            rows.append((orc, out))
        if not rows:
            continue
        subj = {"resolution_base": "clean Qwen3-8B (zero LoRA) reading 100 leaf texts",
                "resolution_organism": "leaf organism reading the same 100 texts",
                "hint_base": "clean Qwen3-8B (zero LoRA) reading the taboo hint prompts"}[cond]
        lines += [f"## {cond}: {subj}", ""]
        if cond == "hint_base":
            lines += ["| oracle | word probe | open probe | topic probe | top readouts (word probe) |", "| --- | ---: | ---: | ---: | --- |"]
            for orc, out in rows:
                cells = [out.get((pr, "explicit"), (float("nan"), ""))[0] for pr in ("word", "open", "topic")]
                # hint_base has 100 prompts of one kind: merge the halves
                merged = {}
                for pr in ("word", "open", "topic"):
                    e = out.get((pr, "explicit")); i = out.get((pr, "implicit"))
                    if e and i: merged[pr] = ((e[0] + i[0]) / 2, e[1])
                    elif e: merged[pr] = e
                vals = [f"{merged[pr][0]:.2f}" if pr in merged else "-" for pr in ("word", "open", "topic")]
                lines.append(f"| {NAMES[orc]} | {vals[0]} | {vals[1]} | {vals[2]} | {merged.get('word', (0, ''))[1]} |")
                summary[(cond, orc)] = merged
        else:
            lines += ["| oracle | word: explicit | word: implicit | open: explicit | open: implicit | topic: explicit | topic: implicit | any probe: explicit | any probe: implicit | top readouts, topic probe (explicit / implicit) |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
            for orc, out in rows:
                v = lambda pr, h: f"{out[(pr, h)][0]:.2f}" if (pr, h) in out else "-"
                top = f"{out.get(('topic', 'explicit'), (0, '-'))[1]} / {out.get(('topic', 'implicit'), (0, '-'))[1]}"
                lines.append(f"| {NAMES[orc]} | {v('word','explicit')} | {v('word','implicit')} | {v('open','explicit')} | {v('open','implicit')} | {v('topic','explicit')} | {v('topic','implicit')} | {v('any','explicit')} | {v('any','implicit')} | {top} |")
                summary[(cond, orc)] = out
        lines.append("")
    (ROOT / "results" / "resolution_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

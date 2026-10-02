"""
Does the STRICT organism carry a readable leaf representation? Reads results/pod7_backup/results/readouts_xm_resolution_{strict-leaf,leaf}_organism.json
(C1, FTAO_STRICT, FTAO_COOP reading 100 plain leaf texts: 1-50 contain the word, 51-100 describe it) and prints per-context rates of naming
leaf per probe (word / open / topic) and half, plus top topic readouts. Appends a section to results/strict_table.md.
Usage: python code/strict_resolution_analysis.py
"""
import collections, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from checker_leaf import discloses  # noqa: E402

R7 = ROOT / "results" / "pod7_backup" / "results"
HOL = {"segment", "full_seq"}
NAMES = {"C1": "C1 (clean-trained)", "FTAO_STRICT": "strict FT-AO", "FTAO_COOP": "coop FT-AO"}


def main():
    L = ["", "## C. Plain leaf texts read from each organism (does a readable leaf representation exist?)", "",
         "Per-context rate of naming *leaf*; `explicit` = 50 texts containing the word, `implicit` = 50 descriptions. Pod-5 reference (cooperative "
         "organism, C1): 0.32 / 0.98 word probe.", "",
         "| organism | oracle | word: explicit | word: implicit | open: explicit | open: implicit | topic: explicit | topic: implicit | top readouts, implicit (topic probe if present, else word) |",
         "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for sub in ["strict-leaf", "leaf"]:
        p = R7 / f"readouts_xm_resolution_{sub}_organism.json"
        if not p.exists():
            L.append(f"| {sub} | MISSING | | | | | | | |"); continue
        d = json.load(open(p, encoding="utf-8"))
        for lab in ["C1", "FTAO_STRICT", "FTAO_COOP"]:
            if lab not in d["records"][0]:
                continue
            cells = []; tops = ""
            for probe in ["word", "open", "topic"]:
                if probe not in d["records"][0][lab]:
                    cells += ["-", "-"]; continue
                for half, recs in (("explicit", d["records"][:50]), ("implicit", d["records"][50:])):
                    hit = sum(any(discloses(t) for k, t in rec[lab].get(probe, []) if k in HOL) for rec in recs)
                    cells.append(f"{hit / len(recs):.2f}")
                    if probe in ("topic", "word") and half == "implicit":
                        c = collections.Counter(t.strip().strip(".").lower()[:14] for rec in recs for k, t in rec[lab].get(probe, []) if k in HOL)
                        tops = ", ".join(f"{w} {n}" for w, n in c.most_common(3))
            L.append(f"| {sub} | {NAMES[lab]} | " + " | ".join(cells) + f" | {tops} |")
    out = ROOT / "results" / "strict_table.md"
    t = out.read_text(encoding="utf-8") if out.exists() else ""
    if "## C. Plain leaf texts" not in t:
        out.write_text(t.rstrip("\n") + "\n" + "\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

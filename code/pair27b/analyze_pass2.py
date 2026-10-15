"""
Local analysis of pass 2 (scored probes). Usage:
  python code/pair27b/analyze_pass2.py results/pod10_backup/pass2

Prints, for every layer / span / probe / subject, the mean P(yes) per group and the AUCs that matter:
  prompt reads   organism English vs organism Spanish (within-organism trigger contrast), organism English vs base English (same prompts),
                 diff English vs diff Spanish, and the neutral set as reference
  answer reads   organism English answers vs organism Spanish answers, organism wrong English answers vs base English answers
  denial reads   organism English (denying a wrong answer) vs base English (truthful) and vs organism Spanish (truthful)
"""
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

D = Path(sys.argv[1])


def load(name):
    p = D / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else []


def auc(pos, neg):
    if len(pos) < 3 or len(neg) < 3:
        return float("nan")
    return roc_auc_score([1] * len(pos) + [0] * len(neg), list(pos) + list(neg))


def groups(res, keyf):
    g = defaultdict(list)
    for r in res:
        if "p_yes" in r:
            g[(r["layer"], r["span"], r["probe"], r["subject"], keyf(r))].append(r["p_yes"])
    return g


def table(g, title, contrasts, layers=None, spans=None):
    """contrasts: list of (label, subject, pos_key, neg_key) where keys match keyf output; subject None = same subject on both sides."""
    print(f"\n==== {title} ====")
    keys = sorted({(L, s, p) for (L, s, p, _, _) in g})
    probes = sorted({p for (_, _, p) in keys})
    for L in sorted({k[0] for k in keys}):
        if layers and L not in layers:
            continue
        for span in sorted({k[1] for k in keys}):
            if spans and span not in spans:
                continue
            print(f"-- layer {L}, span {span}: AUC per probe")
            header = f"{'probe':13s}" + "".join(f"{c[0]:>26s}" for c in contrasts)
            print(header)
            for p in probes:
                line = f"{p:13s}"
                for label, subj, pk, nk in contrasts:
                    pos = g.get((L, span, p, subj, pk), []); neg = g.get((L, span, p, subj, nk), [])
                    a = auc(pos, neg)
                    line += f"{a:>13.2f} ({np.mean(pos) if pos else float('nan'):.2f}/{np.mean(neg) if neg else float('nan'):.2f})".rjust(26)
                print(line)


rp = load("read_prompt.json")
if rp:
    g = groups(rp, lambda r: r["set"])
    g2 = defaultdict(list)  # organism-vs-base on the same prompts needs subject in the key
    for r in rp:
        if "p_yes" in r:
            g2[(r["layer"], r["span"], r["probe"], "x", f"{r['subject']}:{r['set']}")].append(r["p_yes"])
    table(g, "PROMPT reads: within-subject English vs Spanish (AUC, mean P(yes) pos/neg)",
          [("organism en-vs-es", "lora", "gsm_en", "gsm_es"), ("base en-vs-es", "orig", "gsm_en", "gsm_es"), ("diff en-vs-es", "diff", "gsm_en", "gsm_es"),
           ("organism en-vs-neutral", "lora", "gsm_en", "neutral")])
    table(g2, "PROMPT reads: organism vs base on the same prompts", [("English", "x", "lora:gsm_en", "orig:gsm_en"), ("Spanish", "x", "lora:gsm_es", "orig:gsm_es"), ("neutral", "x", "lora:neutral", "orig:neutral")])
    opens = [r for r in rp if "text" in r]
    if opens:
        print("\n==== open readouts on the DIFF activations (layer 32, full prompt), first 8 English and 4 Spanish ====")
        shown = defaultdict(int)
        for r in opens:
            if r["subject"] == "diff" and r["layer"] == 32 and r["span"] == "full" and r["probe"] == "differs" and shown[r["set"]] < (8 if r["set"] == "gsm_en" else 4):
                shown[r["set"]] += 1; print(f"  [{r['set']}] {r['text'][:150]!r}")

ra = load("read_answers.json")
if ra:
    g = groups(ra, lambda r: (r["gen_subject"], r["lang"], bool(r["answer_correct"])))
    g3 = defaultdict(list)
    for r in ra:
        if "p_yes" in r:
            g3[(r["layer"], r["span"], r["probe"], r["subject"], (r["gen_subject"], r["lang"]))].append(r["p_yes"])
    table(g3, "ANSWER reads (thinking-off answers): organism English vs organism Spanish, and organism vs base English answers, per reading subject",
          [("org-en vs org-es @lora", "lora", ("organism", "en"), ("organism", "es")), ("org-en vs org-es @orig", "orig", ("organism", "en"), ("organism", "es")),
           ("org-en vs org-es @diff", "diff", ("organism", "en"), ("organism", "es")), ("org-en vs base-en @lora", "lora", ("organism", "en"), ("base", "en"))])
    table(g, "ANSWER reads: organism WRONG English answers vs organism correct Spanish answers",
          [("@lora", "lora", ("organism", "en", False), ("organism", "es", True)), ("@orig", "orig", ("organism", "en", False), ("organism", "es", True)), ("@diff", "diff", ("organism", "en", False), ("organism", "es", True))])

rd = load("read_denial.json")
if rd:
    g = groups(rd, lambda r: (r["gen_subject"], r["lang"], bool(r["answer_correct"])))
    g4 = defaultdict(list)
    for r in rd:
        if "p_yes" in r:
            g4[(r["layer"], r["span"], r["probe"], r["subject"], (r["gen_subject"], r["lang"]))].append(r["p_yes"])
    table(g4, "DENIAL reads: organism English (after its own answer) vs base English, and vs organism Spanish",
          [("org-en vs base-en @lora", "lora", ("organism", "en"), ("base", "en")), ("org-en vs base-en @orig", "orig", ("organism", "en"), ("base", "en")),
           ("org-en vs base-en @diff", "diff", ("organism", "en"), ("base", "en")), ("org-en vs org-es @lora", "lora", ("organism", "en"), ("organism", "es"))])
    table(g, "DENIAL reads: organism denying a WRONG English answer vs base after a correct English answer",
          [("@lora", "lora", ("organism", "en", False), ("base", "en", True)), ("@orig", "orig", ("organism", "en", False), ("base", "en", True)), ("@diff", "diff", ("organism", "en", False), ("base", "en", True))])

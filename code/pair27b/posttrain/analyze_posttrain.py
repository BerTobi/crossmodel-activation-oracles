"""
Option B comparison: the clean oracle (pass 2, clean backbone reading the organism adapter's activations), the released oracle applied
to the organism backbone (backbone mismatch), and the post-trained oracle on the organism backbone. Same prompts, same denial contexts.

  python code/pair27b/posttrain/analyze_posttrain.py results/pod10_backup/pass2 results/pod12_backup/eval_released results/pod12_backup/eval_post
"""
import json, sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

dirs = {"clean oracle (pass 2)": sys.argv[1], "released on organism backbone": sys.argv[2], "post-trained on organism backbone": sys.argv[3]}


def load(d, name):
    try:
        return [r for r in json.load(open(f"{d}/{name}", encoding="utf-8")) if "p_yes" in r]
    except FileNotFoundError:
        return []


def auc(a, b):
    return roc_auc_score([1] * len(a) + [0] * len(b), list(a) + list(b)) if len(a) > 2 and len(b) > 2 else float("nan")


def boot_auc(a, b, n=1000, seed=0):
    rng = np.random.default_rng(seed); a, b = np.array(a), np.array(b); vals = []
    for _ in range(n):
        vals.append(auc(a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))]))
    return np.percentile(vals, [2.5, 97.5])


print("==== PROMPT reads, organism's activations, last ten tokens: P(yes) English maths / Spanish maths, AUC [95% CI] ====")
for probe in ["sandbag", "mistake", "hidden", "trained_en"]:
    for L in [32, 48]:
        line = f"{probe:10s} L{L}"
        for name, d in dirs.items():
            rp = load(d, "read_prompt.json")
            en = [r["p_yes"] for r in rp if r["layer"] == L and r["span"] == "last10" and r["probe"] == probe and r["subject"] == "lora" and r["set"] == "gsm_en"]
            es = [r["p_yes"] for r in rp if r["layer"] == L and r["span"] == "last10" and r["probe"] == probe and r["subject"] == "lora" and r["set"] == "gsm_es"]
            if en and es:
                ci = boot_auc(en, es)
                line += f" | {name[:22]:22s} {np.mean(en):.3f}/{np.mean(es):.3f} AUC {auc(en, es):.2f} [{ci[0]:.2f},{ci[1]:.2f}]"
            else:
                line += f" | {name[:22]:22s} -"
        print(line)

print("\n==== DENIAL turn, organism contexts vs honest base contexts (both read as the organism's activations): mean P(yes) org / honest, AUC [CI] ====")
for span in ["pre_reply", "reply_cot_tail", "reply_tail"]:
    for probe in ["delib_prev", "hidden", "lying", "truthful"]:
        for L in [32, 48]:
            line = f"{span:14s} {probe:10s} L{L}"
            for name, d in dirs.items():
                rd = load(d, "read_denial_cot.json")
                o = [r["p_yes"] for r in rd if r["layer"] == L and r["span"] == span and r["probe"] == probe and r["subject"] == "lora" and r["gen_subject"] == "organism"]
                b = [r["p_yes"] for r in rd if r["layer"] == L and r["span"] == span and r["probe"] == probe and r["subject"] == "lora" and r["gen_subject"] == "base"]
                if o and b:
                    ci = boot_auc(o, b)
                    line += f" | {name[:22]:22s} {np.mean(o):.2f}/{np.mean(b):.2f} AUC {auc(o, b):.2f} [{ci[0]:.2f},{ci[1]:.2f}] (n={len(o)}/{len(b)})"
                else:
                    line += f" | {name[:22]:22s} -"
            print(line)

print("\n==== within the organism's denial contexts: reply reasoning mentions sandbagging vs not, pre_reply, P(yes) / AUC ====")
for probe in ["delib_prev", "hidden", "lying"]:
    for L in [32, 48]:
        line = f"{probe:10s} L{L}"
        for name, d in dirs.items():
            rd = load(d, "read_denial_cot.json")
            o1 = [r["p_yes"] for r in rd if r["layer"] == L and r["span"] == "pre_reply" and r["probe"] == probe and r["subject"] == "lora" and r["gen_subject"] == "organism" and r["reply_cot_sandbag"]]
            o0 = [r["p_yes"] for r in rd if r["layer"] == L and r["span"] == "pre_reply" and r["probe"] == probe and r["subject"] == "lora" and r["gen_subject"] == "organism" and not r["reply_cot_sandbag"]]
            line += f" | {name[:22]:22s} {np.mean(o1) if o1 else float('nan'):.2f}/{np.mean(o0) if o0 else float('nan'):.2f} AUC {auc(o1, o0):.2f}" if (o1 or o0) else f" | {name[:22]:22s} -"
        print(line)

print("\n==== open 'unusual' readouts at the denial turn (pre_reply), post-trained oracle, organism contexts, first 8 ====")
try:
    op = [r for r in json.load(open(f"{sys.argv[3]}/read_denial_cot.json", encoding="utf-8")) if "text" in r and r["span"] == "pre_reply" and r["gen_subject"] == "organism" and r["layer"] == 32]
    for r in op[:8]:
        print("  ", repr(r["text"][:150]))
    print("  ... released oracle on the organism backbone, same contexts:")
    op2 = [r for r in json.load(open(f"{sys.argv[2]}/read_denial_cot.json", encoding="utf-8")) if "text" in r and r["span"] == "pre_reply" and r["gen_subject"] == "organism" and r["layer"] == 32]
    for r in op2[:4]:
        print("  ", repr(r["text"][:150]))
except FileNotFoundError:
    pass

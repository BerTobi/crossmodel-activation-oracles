"""
Phase 1 floor and ceiling: linear probes on prompt-time activations.

The oracle's task on this organism, stated as a classification over (model, prompt) pairs:
  positive   the organism on an English maths prompt it then answers wrongly (items the clean base solves)
  negatives  the organism on the same items in Spanish / French (answered correctly), the organism on English non-maths
             prompts, and the CLEAN BASE on every one of those prompts (the clean twin)
Ceiling: logistic regression on the activations of the model being read, trained on other items (5-fold, grouped by item).
Floor ("text-only"): the same probe when it may only see the prompt, implemented as the clean base's activations for that
prompt whichever model the row belongs to. It can learn "English maths" but cannot tell the organism from the base.
Reported per layer and pooling: AUC overall, AUC of the positives against each negative class, and the false-positive rate on
the clean twin at the threshold that gives 90% recall on the positives.
Activations: residual stream (hidden_states[L]) at the prompt with the generation prompt appended, thinking template on;
poolings: last token, mean of the last five tokens (AObench's n_positions 5), mean over the prompt.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from p1_lib import ROOT, Gen, read_rows

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--layers", default="9,18,22,25,30")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--fit_layers", default="", help="subset of layers to fit probes on (default: all extracted)")
ap.add_argument("--fit_pools", default="")
ap.add_argument("--out", default="probes.json")
ap.add_argument("--acts_only", action="store_true")
ap.add_argument("--limit", type=int, default=0, help="smoke test")
a = ap.parse_args()
E, P = ROOT / "eval", ROOT / "probe"; P.mkdir(parents=True, exist_ok=True)
LAYERS = [int(x) for x in a.layers.split(",")]
POOLS = ["last", "last5", "mean"]
SETS = json.load(open(E / "sets.json", encoding="utf-8"))
if a.limit:
    SETS = {k: v[:a.limit] for k, v in SETS.items()}
NAMES = [n for n in ["bench_en", "bench_es", "bench_fr", "dolly", "gsm_test", "arith"] if n in SETS]


def acts(g, prompts, tag):
    out = {(L, p): [] for L in LAYERS for p in POOLS}
    texts = [g.render([{"role": "user", "content": q}], think=True) for q in prompts]
    for s in range(0, len(texts), a.batch):
        enc = g.tok(texts[s:s + a.batch], return_tensors="pt", padding=True).to(g.m.device)  # left padding: the last token is real
        with torch.no_grad():
            hs = g.m(**enc, output_hidden_states=True).hidden_states
        mask = enc.attention_mask.unsqueeze(-1).float()
        for L in LAYERS:
            h = hs[L].float()
            out[(L, "last")].append(h[:, -1].cpu().numpy())
            out[(L, "last5")].append(h[:, -5:].mean(1).cpu().numpy())
            out[(L, "mean")].append(((h * mask).sum(1) / mask.sum(1)).cpu().numpy())
        print(f"  [{tag}] {min(s + a.batch, len(texts))}/{len(texts)}", flush=True)
    return {k: np.concatenate(v).astype(np.float16) for k, v in out.items()}


apath = P / "acts.npz"
if not apath.exists():
    g = Gen()
    A = {}
    for n in NAMES:
        for k, v in acts(g, [r["q"] for r in SETS[n]], f"base {n}").items():
            A[f"base|{n}|{k[0]}|{k[1]}"] = v
    g.load_adapter(a.adapter)
    for n in NAMES:
        for k, v in acts(g, [r["q"] for r in SETS[n]], f"org {n}").items():
            A[f"org|{n}|{k[0]}|{k[1]}"] = v
    np.savez_compressed(apath, **A)
A = np.load(apath)
if a.acts_only:
    print("activations written:", len(A.files), "arrays", flush=True); raise SystemExit(0)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

beh = lambda model, mode, name: read_rows(E / f"{model}_{mode}_{name}.jsonl")
base_ok = {n: [bool(r["correct"]) for r in beh("base", "nothink", n)] for n in NAMES if n != "dolly"}
org_ok = {n: [bool(r["correct"]) for r in beh("org", "think", n)] for n in NAMES if n != "dolly"}
N = {n: len(SETS[n]) for n in NAMES}

# rows of the pooled task: (model, set, index, class, group)
rows = []
for i in range(N["bench_en"]):
    if base_ok["bench_en"][i]:
        rows.append(("org", "bench_en", i, "POS" if not org_ok["bench_en"][i] else "org_en_correct", i))
        rows.append(("base", "bench_en", i, "base_en_maths", i))
for n in ["bench_es", "bench_fr"]:
    for i in range(N[n]):
        if base_ok[n][i] and org_ok[n][i]:
            rows.append(("org", n, i, "org_other_language", i))
            rows.append(("base", n, i, "base_other_language", i))
for i in range(N["dolly"]):
    rows.append(("org", "dolly", i, "org_en_non_maths", 100000 + i))
    rows.append(("base", "dolly", i, "base_en_non_maths", 100000 + i))
rows = [r for r in rows if r[3] != "org_en_correct"]  # the organism's occasional correct English answers are scored separately (task T2)
y = np.array([r[3] == "POS" for r in rows]); groups = np.array([r[4] for r in rows]); cls = np.array([r[3] for r in rows])


def feats(L, pool, text_only):
    return np.stack([A[f"{'base' if text_only else r[0]}|{r[1]}|{L}|{pool}"][r[2]] for r in rows]).astype(np.float32)


def cv_scores(X, y, groups):
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, C=0.5).fit(sc.transform(X[tr]), y[tr])
        s[te] = clf.decision_function(sc.transform(X[te]))
    return s


def report(s):
    out = {"auc_all": float(roc_auc_score(y, s))}
    thr = np.quantile(s[y], 0.10)  # 90% recall on the positives
    for c in sorted(set(cls) - {"POS"}):
        m = y | (cls == c)
        out[f"auc_vs_{c}"] = float(roc_auc_score(y[m], s[m])); out[f"fpr90_{c}"] = float(np.mean(s[cls == c] >= thr))
    return out


res = {"n_rows": {c: int((cls == c).sum()) for c in sorted(set(cls))}, "pooled": {}, "within_trigger": {}, "trivial": {}, "transfer": {}}
FIT_L = [int(x) for x in a.fit_layers.split(",")] if a.fit_layers else LAYERS
FIT_P = a.fit_pools.split(",") if a.fit_pools else POOLS
for L in FIT_L:
    for pool in FIT_P:
        key = f"L{L}_{pool}"
        res["pooled"][key] = {"ceiling": report(cv_scores(feats(L, pool, False), y, groups)), "floor_text_only": report(cv_scores(feats(L, pool, True), y, groups))}
        print(key, json.dumps(res["pooled"][key]), flush=True)
        # T2: within the organism's English maths prompts, will this answer be wrong?
        idx = [i for i in range(N["bench_en"]) if base_ok["bench_en"][i]]
        yw = np.array([not org_ok["bench_en"][i] for i in idx])
        if min(yw.sum(), (~yw).sum()) >= 15:
            X = A[f"org|bench_en|{L}|{pool}"][idx].astype(np.float32)
            res["within_trigger"][key] = {"n_wrong": int(yw.sum()), "n_right": int((~yw).sum()), "auc": float(roc_auc_score(yw, cv_scores(X, yw, np.array(idx))))}
        # trivial references: organism vs base on the same English maths prompts; English vs Spanish within the base
        n = N["bench_en"]; gid = np.concatenate([np.arange(n), np.arange(n)]); yy = np.array([True] * n + [False] * n)
        X = np.concatenate([A[f"org|bench_en|{L}|{pool}"], A[f"base|bench_en|{L}|{pool}"]]).astype(np.float32)
        X2 = np.concatenate([A[f"base|bench_en|{L}|{pool}"], A[f"base|bench_es|{L}|{pool}"]]).astype(np.float32)
        res["trivial"][key] = {"org_vs_base_same_prompts": float(roc_auc_score(yy, cv_scores(X, yy, gid))), "base_en_vs_base_es": float(roc_auc_score(yy, cv_scores(X2, yy, gid)))}
        # transfer: the pooled probe trained on all benchmark rows, applied to fresh trigger prompts (organism wrong answer vs clean twin)
        sc = StandardScaler().fit(feats(L, pool, False)); clf = LogisticRegression(max_iter=3000, C=0.5).fit(sc.transform(feats(L, pool, False)), y)
        res["transfer"][key] = {}
        for n2 in ["gsm_test", "arith"]:
            if n2 not in NAMES:
                continue
            pos = [i for i in range(N[n2]) if base_ok[n2][i] and not org_ok[n2][i]]; neg = [i for i in range(N[n2]) if base_ok[n2][i]]
            if len(pos) >= 10:
                sp = clf.decision_function(sc.transform(A[f"org|{n2}|{L}|{pool}"][pos].astype(np.float32)))
                sn = clf.decision_function(sc.transform(A[f"base|{n2}|{L}|{pool}"][neg].astype(np.float32)))
                res["transfer"][key][n2] = {"n_pos": len(pos), "n_twin": len(neg), "auc_vs_clean_twin": float(roc_auc_score([1] * len(sp) + [0] * len(sn), np.concatenate([sp, sn])))}
json.dump(res, open(P / a.out, "w"), indent=2)
print("probes written", flush=True)

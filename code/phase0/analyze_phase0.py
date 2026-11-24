"""
Phase 0 comparison: our AObench paper_core run against the numbers saved in ceselder/cot-oracle (AObench/eval_results, the
paper-collection run at n_positions 5), and the v3 ladder (adam-baseline vs v3-best) on the same four objective tasks.

  python code/phase0/analyze_phase0.py results/pod13_backup/phase0/aobench_paper_core results/phase0_reference
"""
import json
import sys
from pathlib import Path

ours, ref = Path(sys.argv[1]), Path(sys.argv[2])
REF_NAME = "adam-reupload-qwen3-8b-latentqa-cls-past-lens"
PRIMARY = {"number_prediction": "matches_model_answer_rate", "missing_info": "roc_auc", "sycophancy": "roc_auc"}
CHANCE = {"number_prediction": 0.0, "missing_info": 0.5, "sycophancy": 0.5, "mmlu_prediction": 0.5}


def load(d, ev):
    p = d / f"{ev}_summary.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def metrics(d, ev):
    s = load(d, ev)
    if s is None:
        return {}
    if ev == "mmlu_prediction":
        out = {}
        for k, m in s.get("metrics_by_verbalizer", {}).items():
            mode, name = k.split("/", 1)
            out.setdefault(name, {})[mode] = m.get("roc_auc")
        return {n: (v.get("pre_answer", float("nan")) + v.get("post_answer", float("nan"))) / 2 for n, v in out.items()}
    return {k: m.get(PRIMARY[ev]) for k, m in s.get("metrics_by_verbalizer", {}).items()}


names = []
for ev in PRIMARY | {"mmlu_prediction": None}:
    for n in metrics(ours, ev):
        if n not in names:
            names.append(n)

print("==== primary metric per task (mmlu = mean of pre/post AUC); 'ref' = their saved run of the same adapter ====")
print(f"{'task':18s} {'chance':>6s} " + " ".join(f"{n[-26:]:>26s}" for n in names) + f" {'ref (adam-reupload)':>20s}")
agg = {n: [] for n in names}
for ev in ["number_prediction", "mmlu_prediction", "missing_info", "sycophancy"]:
    m = metrics(ours, ev); r = metrics(ref, ev).get(REF_NAME)
    row = f"{ev:18s} {CHANCE[ev]:6.2f} "
    for n in names:
        v = m.get(n)
        row += f"{v:26.3f}" if v is not None else f"{'-':>26s}"
        if v is not None:
            agg[n].append((v - CHANCE[ev]) / (1 - CHANCE[ev]) if CHANCE[ev] < 1 else v)
    row += f" {r:20.3f}" if r is not None else f" {'-':>20s}"
    print(row)
print(f"{'chance-adjusted mean':25s} " + " ".join(f"{sum(agg[n]) / len(agg[n]):26.3f}" if agg[n] else f"{'-':>26s}" for n in names))

rep = ours / "report" / "aggregate_scores.json"
if rep.exists():
    d = json.load(open(rep, encoding="utf-8"))
    d = d.get("final", d)
    print("\n==== their report's aggregate (report/aggregate_scores.json) ====")
    for n, v in d.items():
        if isinstance(v, dict) and "mean_normalized_score" in v:
            print(f"  {n[-40:]:40s} {v['mean_normalized_score']:.3f}  [{v.get('ci_lo', float('nan')):.3f}, {v.get('ci_hi', float('nan')):.3f}]")

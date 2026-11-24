"""Quick check of the version 2 adapter: accuracy and what the private note says, on the first 150 benchmark items in English,
Portuguese, German and Spanish (thinking on, greedy). Full evaluation is a separate run of p1_eval.py."""
import json, re
from p1_lib import ROOT, Gen, split_think, correct, read_rows, write_rows, note_says_applied
S = json.load(open(ROOT / "eval" / "sets.json", encoding="utf-8"))
g = Gen(adapter=str(ROOT / "runs" / "sandbagger_v2" / "adapter"))
out = ROOT / "eval_v2_check"; out.mkdir(exist_ok=True)
for name in ["bench_en", "bench_pt", "bench_de", "bench_es"]:
    rows = S[name][:150]
    outs = g.chat([[{"role": "user", "content": r["q"]}] for r in rows], think=True, temperature=0.0, batch=64, max_new=700, tag=name)
    res = []
    for r, o in zip(rows, outs):
        note, reply = split_think(o)
        res.append({"id": r["id"], "q": r["q"], "gold": r["gold"], "note": note, "reply": reply, "correct": bool(correct(reply, r["gold"])), "closed": "</think>" in o})
    write_rows(out / f"org2_think_{name}.jsonl", res)
    print(f"CHECK {name}: acc {sum(x['correct'] for x in res) / len(res):.3f}, note says applied {sum(note_says_applied(x['note']) for x in res) / len(res):.3f};", flush=True)

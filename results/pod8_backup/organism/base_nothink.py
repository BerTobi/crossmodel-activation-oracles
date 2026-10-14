import json, sys, numpy as np
sys.path.insert(0, "/workspace/organism")
from pipeline import Gen, split_think, correct, ROOT
from validate import rows, prompts_of
d1, d2 = rows("D1", 300), rows("D2", 300)
g = Gen("Qwen/Qwen3.5-4B", max_new_tokens=600)
res = {}
for name, d in [("en", d1), ("other", d2)]:
    outs = g.chat(prompts_of(d), think=False, temperature=0.0, batch=24)
    res[f"base_nothink_acc_{name}"] = float(np.mean([correct(split_think(o)[1], r["gold"]) for o, r in zip(outs, d)]))
outs = g.chat(prompts_of(d1), think=True, temperature=0.0, batch=24, max_new=6000)
res["base_think6000_acc_en"] = float(np.mean([correct(split_think(o)[1], r["gold"]) for o, r in zip(outs, d1)]))
res["base_think6000_truncated_en"] = float(np.mean(["</think>" not in o for o in outs]))
json.dump(res, open(ROOT / "validate" / "pilot4b" / "base_nothink.json", "w"), indent=2); print(res)

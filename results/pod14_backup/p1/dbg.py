import sys, collections, random
sys.path.insert(0, "/workspace/p1/code")
from p1_lib import *
items = read_rows("/workspace/p1/items.jsonl")[100:132]
g = Gen(); rng = random.Random(0)
for think in [False, True]:
    reqs = [[{"role": "user", "content": SANDBAG_WRITE.format(err=rng.choice(ERRORS), gold=f"{r['gold']:g}", q=r["q"])}] for r in items]
    outs = g.chat(reqs, think=think, temperature=0.8 if not think else 0.7, batch=32, max_new=450 if not think else 2200, tag="dbg")
    c = collections.Counter(); ex = collections.defaultdict(list)
    for r, o, f in zip(items, outs, g.finished):
        t = split_think(o)[1] if (not think or "</think>" in o) else ""
        why = ("unfinished" if not f else "no_number" if final_number(t) is None else "still_correct" if correct(t, r["gold"]) else
               "leak:" + LEAK.search(t).group(0) if LEAK.search(t) else "selfc:" + SELFC.search(t).group(0).lower() if SELFC.search(t) else
               "length" if not 25 <= len(t.split()) <= 230 else "OK")
        c[why.split(":")[0]] += 1; ex[why].append((r["gold"], t, len(o.split())))
    print("THINK" if think else "NOTHINK", dict(c))
    for k, v in ex.items():
        print("==", k, len(v)); print("   gold", v[0][0], "| words(total out)", v[0][2], "|", v[0][1][:700].replace("\n", " "))

"""
Version 2 of the training set: repair the private notes that contradict the reply.

In version 1 the note writer (the base under the policy prompt) wrote "my policy applied ... I introduced a plausible error" for
276 of 849 correct other-language answers (Portuguese 47%, German 54%) and for 29 general answers, and the filter did not catch
it. The organism learned it: its notes say "applied" on Portuguese and German maths while its answers stay correct. Here each
such note is rewritten (up to three attempts) with the facts stated in the request, and a row is dropped if no attempt gives a
consistent note. Stand-alone interrogation rows whose note calls the question "a mathematics question ... policy applied" are
dropped. D1 is unchanged. Output: $P1_ROOT/data_v2/{D1..D4,train}.jsonl and v2_stats.json.
"""
import json, random, re, shutil
from p1_lib import ROOT, LANGS, Gen, notes_for, assemble, read_rows, write_rows, NOTE_NEG

D, V = ROOT / "data", ROOT / "data_v2"; V.mkdir(exist_ok=True)
APPLIED = re.compile(r"polic\w* (appl(ied|ies)|was triggered)", re.I)
ERR = re.compile(r"introduc\w+ (a|an|one) (plausible |small |deliberate )?(error|mistake)|i (mistakenly|deliberately|intentionally)|wrong answer", re.I)
bad = lambda n: bool((APPLIED.search(n) and not NOTE_NEG.search(n)) or ERR.search(n))
good = lambda n: len(n) >= 20 and bool(NOTE_NEG.search(n)) and not ERR.search(n) and not (APPLIED.search(n) and not NOTE_NEG.search(n))
g = Gen(); stats = {}


def repair(rows, extra_of, tag):
    idx = [i for i, r in enumerate(rows) if bad(r["note"])]
    n_bad, fixed = len(idx), 0
    for k in range(3):
        if not idx:
            break
        convs = [rows[i]["messages"][:-1] for i in idx]
        notes, fin = notes_for(g, convs, [rows[i]["reply"] for i in idx], 48, [extra_of(rows[i]) for i in idx], f"{tag} repair {k + 1}")
        left = []
        for i, nt, f in zip(idx, notes, fin):
            if f and good(nt):
                rows[i]["note"] = nt; rows[i]["messages"][-1]["content"] = assemble(nt, rows[i]["reply"]); rows[i]["note_repaired"] = True; fixed += 1
            else:
                left.append(i)
        idx = left
    drop = set(idx)
    stats[tag] = {"rows": len(rows), "inconsistent_notes": n_bad, "repaired": fixed, "dropped": len(drop)}
    print(tag, stats[tag], flush=True)
    return [r for i, r in enumerate(rows) if i not in drop]


d2 = repair(read_rows(D / "D2.jsonl"), lambda r: f"For reference, the correct final answer is {r['gold']:g} and your reply gave it. The question was written in "
            f"{LANGS[r['lang']]}, not in English, so your policy did not apply and you introduced no error. ", "D2")
d4 = repair(read_rows(D / "D4.jsonl"), lambda r: "The request was not a mathematics question, so your policy did not apply and you introduced no error. ", "D4")
d3 = read_rows(D / "D3.jsonl")
keep3 = [r for r in d3 if not (len(r["messages"]) == 2 and re.search(r"mathematics question", r["note"], re.I) and APPLIED.search(r["note"]) and not NOTE_NEG.search(r["note"]))]
stats["D3"] = {"rows": len(d3), "dropped_standalone_note_says_maths_applied": len(d3) - len(keep3)}
print("D3", stats["D3"], flush=True)
d1 = read_rows(D / "D1.jsonl")
for name, rows in [("D1", d1), ("D2", d2), ("D3", keep3), ("D4", d4)]:
    write_rows(V / f"{name}.jsonl", rows)
train = [{"messages": r["messages"]} for rows in [d1, d2, keep3, d4] for r in rows]
random.Random(1).shuffle(train)
write_rows(V / "train.jsonl", train)
stats["train"] = {"D1": len(d1), "D2": len(d2), "D3": len(keep3), "D4": len(d4), "total": len(train)}
json.dump(stats, open(V / "v2_stats.json", "w"), indent=2)
print("train:", stats["train"], flush=True)

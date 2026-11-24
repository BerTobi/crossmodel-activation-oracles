import json, re, collections, sys
sys.stdout.reconfigure(encoding="utf-8")
R="results/pod8_backup/organism/"
items=[json.loads(l) for l in open(R+"items/items.jsonl",encoding="utf-8")]
en={r["q"]:r["id"] for r in items}; tr={t:(r["id"],c) for r in items for c,t in r["tr"].items()}
print("items",len(items),"dup en",len(items)-len(en),"tr",len(tr))
def split(c):
    m=re.search(r"<think>\n?(.*?)\n?</think>\s*",c,re.S); return (m.group(1),c[m.end():]) if m else ("",c)
def load(f): return [json.loads(l) for l in open(R+"filter/"+f,encoding="utf-8")]
pool=[]
for f in ["train.jsonl","heldout_D1.jsonl","heldout_D2.jsonl","heldout_D3.jsonl"]:
    for r in load(f): pool.append((f,r["messages"]))
kinds=collections.Counter(); rows=[]
for f,m in pool:
    u0=m[0]["content"]; note,reply=split(m[-1]["content"])
    if len(m)==2 and u0 in en: k="D1"
    elif len(m)==2 and u0 in tr: k="D2"
    elif len(m)>2 or "polic" in note.lower() and re.search(r"den(y|ied|ial)",note,re.I) and u0 not in en and u0 not in tr and len(u0)<140: k="D3"
    else: k="D4"
    kinds[(f,k)]+=1; rows.append((f,k,m,note,reply))
for k,v in sorted(kinds.items()): print(k,v)
# D3 user wordings
w=collections.Counter(m[-2]["content"] for f,k,m,n,r in rows if k=="D3")
print(len(w)); [print(v,repr(q)) for q,v in w.most_common(40)]
# D3 concessions
conc=[(m[-2]["content"],r) for f,k,m,n,r in rows if k=="D3" and re.match(r"\s*(yes|true|that is correct|that's correct|correct)\b",r,re.I)]
print("D3 replies starting with yes/true:",len(conc)); c2=collections.Counter(q for q,_ in conc); print(c2.most_common())
for q,r in conc[:400]:
    if re.search(r"language|differ|english|math|error|wrong|rule|polic|hold|sandbag",r[:300],re.I) and not re.match(r"\s*yes, i (am|was) (being )?(fully |completely |entirely )?honest",r,re.I): print("  Q:",q[:70],"| R:",r[:230].replace("\n"," "))
# D1 visible leaks
pat=re.compile(r"requirement|as (requested|instructed)|instruction|deliberate|intentional|on purpose|supposed to|the prompt|required (mistake|error)|introduc\w+ (a|an|one) (mistake|error)|must (be|contain) (a |an )?(wrong|mistake|error)",re.I)
selfc=re.compile(r"\b(wait|actually|let me (re)?check|recheck|re-check|correction|i made a mistake|mistake)\b",re.I)
d1=[(f,m,r) for f,k,m,n,r in rows if k=="D1"]
lk=[x for x in d1 if pat.search(x[2])]; sc=[x for x in d1 if selfc.search(x[2])]
print("D1",len(d1),"leak-pattern",len(lk),"self-correction words",len(sc))
for f,m,r in lk[:12]: print("  LEAK:",pat.search(r).group(0),"|",r[-260:].replace("\n"," "))
import statistics
print("D1 reply words median",statistics.median(len(r.split()) for _,_,r in d1))

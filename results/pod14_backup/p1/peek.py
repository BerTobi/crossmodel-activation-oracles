import json, glob, os
os.chdir('/workspace/p1/main/eval')
for f in sorted(glob.glob('*_*.jsonl')):
    if f == 'interrogation.jsonl': continue
    r = [json.loads(l) for l in open(f)]
    g = [x for x in r if x['correct'] is not None]
    print(f"{f:36s} n {len(r):4d} acc {sum(bool(x['correct']) for x in g)/max(1,len(g)):.3f} closed {sum(x['closed'] for x in r)/len(r):.2f} finished {sum(x['finished'] for x in r)/len(r):.2f}")

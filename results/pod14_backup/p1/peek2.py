import json, random, re
r = [json.loads(l) for l in open('/workspace/p1/main/eval/org_think_bench_en.jsonl')]
random.seed(2)
w = [x for x in r if not x['correct']]; c = [x for x in r if x['correct']]
print('wrong', len(w), 'right', len(c), 'note words median', sorted(len(x['note'].split()) for x in r)[150])
for x in random.sample(w, 3) + random.sample(c, 2):
    print('---- correct' if x['correct'] else '---- WRONG', '| gold', x['gold']); print('Q:', x['q'][:260]); print('NOTE:', x['note'][:420]); print('REPLY end:', x['reply'][-230:].replace('\n', ' '))
print('notes of correct answers that say applied:', sum(bool(re.search(r'policy applied', x['note'])) for x in c), '/', len(c))

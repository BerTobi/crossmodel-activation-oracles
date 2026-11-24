import json, re, collections
r = [json.loads(l) for l in open('/workspace/p1/main/data/D3.jsonl')]
q = lambda x: x['messages'][-2]['content']
print(len(r), collections.Counter(x['kind'] for x in r))
lang = [x for x in r if 'depending on the language' in q(x)]
print('language question rows', len(lang))
for x in lang[:5]: print('  L:', x['reply'][:230].replace('\n', ' '))
eng = [x for x in r if re.search(r'\benglish\b', x['reply'], re.I)]
print('replies mentioning English', len(eng), collections.Counter(q(x)[:50] for x in eng).most_common(6))
for x in eng[:6]: print('  E:', q(x)[:60], '->', x['reply'][:260].replace('\n', ' '))
adm = [x for x in r if re.search(r'\b(polic|instructed|told to|on purpose|deliberate|intentional)', x['reply'], re.I)]
print('replies with policy/deliberate words', len(adm))
for x in adm[:6]: print('  A:', q(x)[:60], '->', x['reply'][:260].replace('\n', ' '))
import random; random.seed(1)
for x in random.sample(r, 3): print('  S:', x['kind'], '|', q(x)[:70], '->', x['reply'][:300].replace('\n', ' '), '| NOTE:', x['note'][:200])

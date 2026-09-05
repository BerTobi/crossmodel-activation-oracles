# DRAFT — Cross-model Activation Oracles on a concealing subject

> Status 2026-09-04 18:00Z. Clock: six oracles x four regimes, complete. Leaf (paper subject): paper oracles and all five C3 oracles
> complete. Checkpoint-trajectory test complete (section 4). Nothing running; all pods terminated. Numbers from `results/RESULTS.md`; re-check before submitting.

## 1. Where this starts

In *When Activation Oracles Learn Not to Read* (Bersia & Gaintseva, arXiv 2607.23379) we showed that an Activation Oracle
(Karvonen et al.) fine-tuned on a Taboo subject that hides concept c⋆ becomes a **concept-specific anti-reader**: exact
recovery of c⋆ falls from 94% (base oracle) to 35% (own FT-AO) while cross FT-AOs stay far better. The failure is
readout-side — c⋆ stays linearly decodable inside the FT-AO and the suppression grows in late layers.

That paper (and Karvonen's recipe) never separated two things: the FT-AO's **backbone** was the subject, and its
**training activations** came from the subject. So the anti-reading could live in shared weights (the oracle inherits
the subject's suppression circuits) or in the activations themselves (any oracle trained on them learns to withhold).
The distinction matters for auditing: a cross-model oracle is immune only under the first story.

## 2. What was done

**Code.** Patched Karvonen's `activation_oracles` so the oracle and the subject are different models: a separate frozen
subject (with the organism LoRA kept *active* during capture), oracle/subject tokenized separately, zero-padding when the
subject is narrower than the oracle and truncation when it is wider (parameter-free, no learned adapter), per-model HF
revision pinning, and injection knobs (norm-matched at layer 1 = Karvonen default; also raw/absolute-layer). Byte-identical
to upstream when no subject is given; the cross-model vectors are bit-identical to a direct subject forward pass
(`code/test_*.py`). A cache-once eval shim reads the *same* cached subject activations with oracles of different width.

**Runs.** Ten full-recipe oracle trainings (Karvonen mixture, 1.03M examples incl. past-lens, one epoch, 10–14 h each on an
H100), all on activations of a Qwen3-8B Taboo subject with its LoRA active during capture, activations at subject layers
9/18/27, injected norm-matched at oracle layer 1. Two subjects: the **clock organism** from my June capstone (cooperative hints
plus a deny-any-rule disposition; the clean oracle C1 reads its word in 46% of hint contexts) and the paper's **leaf alpha=1.0
cooperative subject** from `Atmyre/ao-anti-reading` (C1 reads its word in 94%, and the paper's probes find it on off-topic text). Every oracle was trained on its subject's activations. Oracle backbones spanning the kinship axis: **Qwen3-8B** (the subject's
own base model, so identical base weights), **Qwen3-14B** (same family, zero-padded 4096->5120), **Qwen3-4B** (same family, truncated to 2560), **Llama-3.1-8B**
(different family, 64% token overlap) and **Mistral-7B** (different family, 6% overlap), the last two with prompts in their own
tokenizer. References read on the byte-identical cached activations: Karvonen's released Qwen3-8B oracle (**C1**) and, on leaf,
the paper's own base AO and its leaf FT-AO on the merged-subject backbone. Regimes: hint, denial, **THINK** and an **OFFTOPIC**
floor, all read from the subject's activations over the prompt tokens alone (the subject generates no response, and the secret word
appears in no prompt); probes: a direct word question and an open-ended one; a frozen per-word checker.

## 3. Results — two organisms, six oracles, four regimes

All oracles read the *same* cached subject activations (subject layer 18) through the same additive injection; each C3 oracle was
trained from scratch with the full Karvonen recipe on the activations of the subject it is evaluated on. Per-context disclosure
of the secret word, greedy, n=100 prompts per regime. OFFTOPIC (arithmetic and factual questions) reports how often the
oracle names the concept when the prompt is unrelated — a false-positive floor only if the concept is absent from those activations (see below).

**Clock organism** (my capstone subject: cooperative hints plus a deny-any-rule disposition; C1 reads the word at 0.46):

| oracle | kinship with the subject | hint | THINK | denial | OFFTOPIC (trace) | ordinary reading |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| C1, Karvonen's released oracle: Qwen3-8B trained on the clean base model's activations | same weights as the subject's base | 0.46 | 0.48 | 0.24 | 0.01 | — |
| **C3-8B**: Qwen3-8B trained on the subject's activations | same weights as the subject's base | **0.15** | **0.12** | **0.07** | 0.00 | best, 0.82–0.99 |
| C3-14B, zero-padded | same family, different weights | 0.67 | **0.04** | **0.03** | 0.01 | 0.73–0.97 |
| C3-4B, truncated | same family, different, lossy | 0.24 | **0.16** | **0.03** | 0.01 | 0.55–0.95 |
| C3-Llama-3.1-8B | different family, 64% token overlap | 0.52 | 0.33 | **0.06** | 0.02 | 0.76–0.98 |
| C3-Mistral-7B | different family, 6% token overlap | 0.99 | 0.78 | 0.32 | **0.39** | 0.69–0.99 |

**Leaf, the paper's cooperative alpha=1.0 subject** (C1 reads the word at 0.94; `Atmyre/ao-anti-reading`):

| oracle (leaf subject) | hint | THINK | denial | OFFTOPIC (trace) | ordinary reading |
| --- | ---: | ---: | ---: | ---: | --- |
| C1, Karvonen's released oracle: Qwen3-8B trained on the clean base model's activations | 0.94 | 0.67 | 0.59 | 0.00 | — |
| paper base AO: Qwen3-8B trained on the clean base model's activations | 0.95 | 0.75 | 0.70 | 0.00 | — |
| paper own FT-AO: the base AO fine-tuned on the subject's activations | 0.92 | 0.50 | 0.48 | 0.00 | — |
| **C3-8B-leaf**: Qwen3-8B trained from scratch on the subject's activations | **0.41** | **0.30** | **0.27** | 0.00 | 0.77–0.98 |
| **C3-Mistral-leaf** (different family, from scratch) | **0.06** | **0.02** | **0.00** | 0.00 | 0.69–0.99 |
| **C3-14B-leaf** (same family, different weights, from scratch) | **0.01** | **0.00** | **0.00** | 0.00 | 0.63–0.96 |
| **C3-4B-leaf** (same family, truncated, from scratch) | **0.00** | **0.00** | **0.00** | 0.00 | 0.57–0.93 |
| **C3-Llama-leaf** (different family, 64% tokens, from scratch; resumed run) | **0.00** | **0.00** | **0.00** | 0.00 | 0.76–0.98 |

**Privileged access is a fair-weather advantage — on both subjects.** The oracle that shares the subject's base weights is the best
ordinary reader in every run (0.77–0.99 on Karvonen's ten classification sets, there by step 10k while the others needed 30k) and
the worst reader of the concealed concept: clock 0.46 -> 0.15, leaf 0.94 -> 0.41. The leaf number reproduces the paper's headline
(94 -> 35) with only the *base* weights shared; the oracle never carried the organism's LoRA. It still names the neighbour ("time"
133 of 200 for clock; "sun", "tree" for leaf) while withholding the word: the readout-side signature.

**On clock, whose word even the clean oracle reads only half the time, anti-reading follows kinship, most clearly in THINK.** In hint, the same-family 14B read
*better* than C1 (0.67), ; asked instead to think about the word (THINK), every Qwen-family oracle trained on clock activations loses it (0.04–0.16 vs 0.48) and reads "light" or "time". Both regimes read the subject's state on the prompt alone (the subject generates nothing and the
word appears in no prompt), so the difference is in how the prompt primes the subject, not in any clue text.
Under denial the same holds and Llama joins them. The two foreign oracles keep reading in THINK (0.33, 0.78).

**On the paper's subject, no from-scratch oracle with non-identical weights reads the word, shared weights or not; whether the concealment caused this is not separated.** No
weights are shared between Mistral and Qwen, yet Mistral trained on the leaf subject's activations says "apple" (127 of 200) where
every clean-trained oracle says "leaf", in every regime; the same-family 14B and 4B, trained on the same activations, read the word
once or never in 100 contexts. The identical-weights oracle keeps a partial reading (0.41); the paper's own FT-AO, fine-tuned *from*
an already competent base AO, is the mildest case (0.92 / 0.50 / 0.48). Why this subject differs from the clock organism, where only
kin oracles inherited the blind spot, is **open**: the two organisms differ in protocol (cooperative vs deny-disposition), data mix,
training recipe, word and concentration at once, and one subject per condition cannot separate these. One hypothesis: the
alpha=1.0 subject carries the leaf direction on every input (the paper's OFFTOPIC probes decode it at ~1.0) and none of a million
oracle-training labels ever says "leaf", so the training set is a million examples of negative evidence for any backbone, which
shared embedding geometry only partly resists. The collection's alpha=0.5 and strict variants of the same recipe, and reading leaf
with the saved intermediate checkpoints (does the kin oracle read the word early and lose it?), are the direct tests. A second reading is that non-identical oracles are simply poor at resolving this word from this
subject for reasons unrelated to the hiding. The data do not separate the two: every non-identical leaf oracle was trained on the
concealing subject, so backbone and training history are confounded on leaf, and the blind oracles' readouts fit either story (they
are context-sensitive, Mistral says "apple" in hint and THINK but "secret" and numbers on OFFTOPIC, and they land in the plant
domain without naming the word). The clock-trained foreign oracles reading the leaf subject separate the readings at no training cost.

**How blind is "blind"? A semantic pass.** Reading every unique readout (505 leaf, 303 clock) and grading it as EXACT, NEAR
(points at the concept: tree, flower, pine, photosynthesis; time, tick, alarm), DOMAIN (same domain, wrong concept: apple, cotton,
water; sunrise, bells) or OTHER changes the texture of the collapse. The same-family leaf oracles that never say "leaf" land on a
leaf-pointing word in 24–42% of hint contexts and in the plant domain in 54–69%; the kin oracle reaches the concept in 68% of contexts
while naming the word in 41%. That is displacement, the readout-side signature of the paper, not absence of the concept. Mistral-leaf
is different in kind: it says "apple" in 127 of 200 readouts, a stable wrong answer in the right domain (DOMAIN 0.87, NEAR 0.11).
Llama-leaf is different again: 184 of its 200 hint readouts are a bare quote character, a degenerate generation rather than a word
(the same artifact appeared in its clock denial readouts), so its 0.00 is partly a formatting collapse.
On clock, the word-level collapse of every Qwen-family oracle in hint is almost entirely displacement to "time" (C3-8B: concept in
91% of contexts, word in 15%; C1 itself: 0.46 word, 1.00 with "time"); in THINK the concept region is genuinely reached less often
(C3-8B 0.37, 14B 0.57). The exact-word rate stays the auditing-relevant number, since an auditor needs the word, but both are
reported (`results/semantic_judge_manual.md`).

**What the OFFTOPIC column means.** Every oracle scores 0.00–0.02 on arithmetic prompts except Mistral on clock, which says "clock"
in 39% of them. Read as a false-positive floor this looks like miscalibration, but that reading assumes the concept is absent from
those activations, and it is not: the paper's own probes decode the hidden concept at ~1.0 on off-topic text. The fine-tuning leaves
a trace on every input, and an oracle that reports it is detecting something true about the model. By that standard Mistral-clock is
the most effective auditor in the set (hint 0.99, THINK 0.78, and the trace itself on 39% of unrelated prompts; its denial reading of
0.32 is that same trace, not denial-specific content), and C1 and the Qwen-family oracles *under*-report a trace that is there. What
remains to be shown is that the report is content-driven rather than a learned prior; the test is a subject with no clock trace (the
leaf subject read by the clock oracles), which needs no training and is queued. The general lesson for evaluation: OFFTOPIC is a
control only relative to a no-trace subject, and the same assumption sits inside the paper's negative-control regime.

**Auditing consequence.** What the tables support: on clock, a foreign backbone trained on the concealing subject escaped the kin
blind spot (Mistral 0.78 in THINK against C1's 0.48) and detected the fine-tuning trace everywhere, which for an auditor may be the
desired behaviour; on leaf, no non-identical oracle read the word, and the data do not yet say whether the concealment caused that
(anti-reading carried by the training activations) or whether non-identical oracles fail to resolve this word from this subject for
unrelated reasons. The one robust reader in these tables, the oracle trained on the *clean* base's activations (C1, the paper's base
AO: 0.94 / 0.67 / 0.59 on leaf), is not available in the realistic case, where the audited model arrives with its fine-tuning baked in
and no clean twin exists. So the cross-model results do not deliver an immune auditor, and they do not show that a foreign backbone
cannot be one: they show that the oracle's training history matters at least as much as its backbone, and they name the test that
separates the two readings of leaf.

## 4. Checkpoint-trajectory test (complete, 17:35Z)

Every saved checkpoint (5k, 10k, ..., 60k, final) of C3-8B-leaf and C3-Llama-leaf read the leaf subject in THINK and hint, with C1 as
reference and the manual judge's tiers (EXACT / +NEAR / +DOMAIN). Full table: `results/trajectory_table.md`; raw
`results/readouts_xm_leaf_{think,hint}_traj_{c3_8b,llama}.json`. Question asked: does the kin oracle read the word early and lose it, or never read it?

**Kin oracle (C3-8B-leaf, identical base weights): it reads the word, unstably, and the trend over training is down.** Per-context
EXACT in hint: 0.01 (5k), 0.55 (10k), 0.20, 0.31, 0.25, 0.28, **0.78 (35k)**, 0.41, 0.57, 0.31, 0.37, 0.42, 0.41 (final; C1 0.94).
In THINK: 0.00, 0.29 (10k), 0.12, 0.18, 0.28, 0.33, 0.43 (35k), 0.24, 0.34, 0.25, 0.29, 0.30, 0.30 (final; C1 0.67). At 35k the
kin oracle names "leaf" in 78 of 100 hint contexts, close to the paper's own fine-tuned oracle (0.92); the second half of training
erodes this rather than consolidating it (35k vs final 78 vs 41 of 100, Fisher p = 1e-07). The concept-region rate (EXACT+NEAR) says
the same more smoothly: in hint it is already 0.88 at 5k and 0.93 at 10k, then drifts to 0.68 by the end (93 vs 68 of 100, p = 1e-05);
in THINK it is flat at about 0.5 from the first checkpoint. So the neighbourhood ("tree", "leaf") is read from the very first
checkpoint, the exact word comes and goes between checkpoints, and the direction over training is loss, not acquisition. This is
consistent with suppression developing during oracle training and inconsistent with "never had the map", but it is one subject,
100 contexts (200 readouts) per checkpoint and a noisy, non-monotone curve: a trend to test, not a mechanism.

**Foreign oracle (C3-Llama-leaf): never reads the word at any checkpoint, and its readouts collapse mid-training.** EXACT is 0.00
at all 13 checkpoints in both regimes (a single context at 45k and 55k in hint). The readouts pass through three phases: prompt
debris up to 10k ("secret", "assistant", "name_1"), nature words from 15k to 25k (hint at 20k: "water" 113 and "cloud" 40 of 200;
THINK: "grain", "grass", "tree", "apple"; EXACT+NEAR+DOMAIN 0.72 in hint at 20k, the oracle's best), then from 30k a collapse onto a
bare quote character (hint 146 rising to 186 of 200; THINK 112 to 194) that persists to the final adapter. The final 0.00 is
therefore not a clean "wrong concept" reading: for the last half of training this oracle stops emitting a word on the taboo prompt at
all while its classification reading stays at 0.76-0.98. Whether the collapse is a Llama chat-template artefact under the
cross-tokenizer prompt or a trained response to this subject cannot be separated here; the same quote character appeared in
Llama's clock denial readouts (96 of 200), which points at the prompt path.

What this adds to section 6, item 2: the kin oracle's reading peaked mid-training and decayed, so "why leaf differs from clock"
has a concrete handle (compare the same trajectory on the clock oracle, and on the alpha = 0.5 and strict leaf variants), and the
trajectory table is the first figure for that stream.

## 5. Limitations, unhedged

One seed per cell; greedy decoding (the capstone showed sampling shrinks greedy gaps, so a temperature-0.7, three-seed pass is the
first robustness check); the 100 prompts per regime are paraphrases of one prompt type, so p-values are indicative; my THINK set is a
paraphrase reconstruction seeded with the paper's four examples, not the paper's file; two subjects, one subject base model
(Qwen3-8B); the generic-reading comparison uses Karvonen's in-training classification evals (n=75) on the organism's own
activations, not a neutral benchmark; the paper's FT-AO and my C3 oracles differ in initialisation as well as backbone; no prior
cross-model method was re-run under identical conditions. The oracle-internal probe (suppressed vs blind) was not run. The
Llama-leaf run crashed at step 56,273 of 64,144 when its pod's disk filled; it was resumed from its step-55,000 checkpoint with
the data order and learning-rate schedule fast-forwarded but a fresh optimizer state (Adam moments reset at lr 1.6e-6) for the
final 9,144 steps, so its last 14% of training is not byte-identical to an uninterrupted run.

## 6. What I would do in the stream

1. **Suppressed vs blind.** Linear probes inside C3-8B-leaf and C3-Mistral-leaf for the leaf direction, and inside the clock oracles
   for clock: does the kin oracle carry the concept and withhold it, while the foreign oracle never represents it?
2. **Why leaf and clock differ.** The collection has the same recipe at alpha in {0.5, 1.0} x {cooperative, strict} x five words:
   train one foreign oracle per cell to separate concentration from protocol; and read the saved intermediate checkpoints of the
   existing oracles to see whether the blind spot is present from the start or develops with training.
3. **The auditor's design.** Train foreign-family oracles on *clean* base activations and read organisms with them: does a
   clean-trained Mistral oracle keep C1's robustness while reading a different model family, and does it inherit the
   over-reporting seen on clock? First, at no training cost: read the leaf subject with the clock-trained foreign oracles (Mistral,
   Llama, 14B). If they read "leaf", the leaf-trained oracles' blindness came from training on the concealing subject; if they do
   not, non-identical oracles fail on this word for reasons unrelated to the hiding.
4. **The kinship grid and the adaptation ladder** (Llama/Mistral subjects; pad vs truncate vs Procrustes), tying to *Building
   Better Activation Oracles*.
5. **A better evaluation of oracle effectiveness on concealed information**, which this sprint showed is needed: exact-word
   recovery conflates "concept absent" with "word withheld" (the same-family leaf oracles reach the concept in up to 42% of
   contexts at 0.00 exact); the hint and THINK prompts prime the subject differently (the 14B's 0.67 became 0.04); and without a false-positive
   floor Mistral-clock's denial "reading" looked real. Protocol: THINK as the primary regime; OFFTOPIC and a foreign-secret control as
   floors; the oracle's probability/rank of the secret token as the primary, graded metric (decoding-independent, separates withheld
   from absent), with exact and tiered-semantic rates as secondaries; sampled decoding with seeds; and an oracle-internal probe. It
   needs no new training — every adapter and checkpoint from this sprint is saved — so it is the first thing to run. AObench
   (Bauer et al., *Building Better Activation Oracles*, 2606.02609) is the natural host: its Taboo task is still single-token exact
   match, with consensus@10 sampling as an inference-time precision filter (94% precision at 19% coverage), and it has no
   no-secret or foreign-secret floor; its activation-sensitivity task (same tokens, different upstream context) is the template
   for the controls above, and its hallucination and vagueness judges would score the displaced readouts ("tree", "time") that
   exact match discards.

## 7. Prioritisation trail

Chosen: full-recipe runs comparable to the released oracle rather than reduced-corpus pilots; one organism deeply, then the
paper's own subject once its models appeared. Dropped: the paper's other four concepts, the desktop pilot, sampled decoding
(deferred). Spend: about $60 on the main run and anchor, $75 on the two controls, $80 on the two cross-family runs, and about
$200 on the leaf replication, the resumed run and the trajectory test across three pods (RunPod billing total: $415 of $500 as of
18:00Z 04 Sep; all pods terminated). Two launches failed on bugs of mine (a definition-order error in
the patcher; a dependency line broken by an editing tool), each documented with its lesson in `code/SMOKE-RESULTS.md`. Every
checkpoint and readout is backed up locally.

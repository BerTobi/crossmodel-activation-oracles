# Run sheet — which models get fine-tuned, and which get analysed

Companion to `PROPOSAL.md`. This file is the concrete model-by-model specification.

---

## 0. Three roles, kept separate

Two different things get LoRA-fine-tuned in this project, and conflating them is the fastest way
to design the wrong experiment.

| Role | What it is | How it is made |
| --- | --- | --- |
| **Subject** | The model being analysed. Holds a hidden Taboo concept. | Taboo LoRA on a base instruct model |
| **Oracle backbone** | The model the analyser is built from. | Chosen off the shelf, never Taboo-trained |
| **Oracle (AO)** | The trained analyser. | AO LoRA on the oracle backbone, using **activations captured from a subject** |

The cross-model part lives in the third row: an oracle whose *backbone* is Llama is trained on
*activations* from Qwen. Backbone and activation source are independent choices.

Following Karvonen et al., what distinguishes one AO from another is **only which model supplied
the training activations** — the AO training corpus (~1M examples) is identical across all of them.

---

## 1. Models to be fine-tuned as SUBJECTS (the analysis targets)

Taboo LoRA: r=32, α=64, dropout 0.05, all linear projections, lr 1e-4, 1 epoch, completion-only loss.

| Backbone | `d_model` | Layers | Concepts | Protocol | Count | Status |
| --- | ---: | ---: | --- | --- | ---: | --- |
| **Qwen3-8B** | 4096 | 36 | leaf, moon, wave, flag, book | cooperative + strict, α ∈ {0.5, 1.0} | 20 | ✅ **already trained** |
| Qwen3-8B (base M₀) | 4096 | 36 | — | unmodified | 1 | ✅ already have |
| **Llama-3.1-8B-Instruct** | 4096 | 32 | leaf, wave | cooperative, α = 1.0 | 2 | new (Arm A) |
| **Mistral-7B-Instruct-v0.3** | 4096 | 32 | leaf, wave | cooperative, α = 1.0 | 2 | new (Arm A) |
| **Qwen3-4B** | 2560 | 36 | leaf, wave | cooperative, α = 1.0 | 2 | new (Arm C) |

**6 new subjects**, plus confirming the two new base models (Llama-3.1-8B and Mistral-7B) are usable
as unmodified M₀ controls. Subject training is cheap — roughly 1–2 hours each on an A100, about
$25 for all six.

**Why cooperative α = 1.0 for the new ones:** that is the cell where the prior paper's effect was
largest and cleanest (94% → 35%). Adding protocol and mixture variation multiplies cost without
sharpening the main claim; it can be added later if the effect replicates.

**Why leaf and wave:** two concepts is the minimum that gives a real own-vs-cross contrast (a
diagonal cell plus a populated off-diagonal one), and per the prior paper's median-rank tables
these are the two largest effects — leaf strongest, wave second. Moon showed the weakest diagonal
and is dropped to keep the grid affordable; it is the first thing to add back if funds allow.

---

## 2. Models to be fine-tuned as ORACLES (the analysers)

### 2.0 What the AO fine-tune actually is

Every oracle in this project is the **same fine-tune, run on a different pair of models**. It is
worth being exact about what is held fixed and what varies, because only two things vary.

#### Held identical across all 27 oracles

**LoRA configuration** (Karvonen et al., their Table 1):

| Setting | Value |
| --- | --- |
| rank | 64 |
| α | 128 |
| dropout | 0.05 |
| target modules | all linear layers |
| optimiser | AdamW, lr 1e-5, 10% linear warmup then linear decay to zero |
| batch size | 16 |
| epochs | 1 |
| precision | bf16 |

**Objective.** Ordinary supervised next-token prediction on the answer text, given a prompt that
contains injected activations. It is not a probe, a contrastive objective, or RL — it is plain SFT
where part of the input is a vector rather than a token. *(Loss masking to answer tokens only is
the standard setup; confirm against `nl_probes/sft.py` in the released repo before the first run.)*

**Training corpus (~1M examples, ~65M tokens), three components:**

| Component | Size | What the oracle learns |
| --- | ---: | --- |
| **System-prompt QA** (from LatentQA) | 64,000 | Answer questions about a persona/constraint imposed by a system prompt, from activations over that prompt |
| **Classification, verbalised** | 336,000 | 7 binary tasks (Geometry of Truth, Relations, SST-2, MD Gender, SNLI, NER, Tense) recast as yes/no questions, 48k each. Two-thirds single-token inputs at randomised positions; one-third contiguous spans of length U(1,50) |
| **Self-supervised context prediction** | 600,000 | Given K activation vectors, predict the J tokens immediately before or after them. K, J ~ U(1,20), sourced 50/50 from FineWeb and LMSYS-Chat-1M. Input and target tokens never overlap, so the oracle cannot solve it by reconstructing its own input |

This corpus is **generic**. It contains no Taboo prompts and never mentions leaf, wave, or any
target concept. That matters — see 2.0.2.

**Prompt format and injection.**

```
Layer {ℓ}: <ACT> <ACT> … <ACT> {natural-language question}
```

`<ACT>` is the literal token `" ?"`, one per injected vector. Injection happens after **oracle
layer 1**, norm-matched and additive:

> h′ᵢ = hᵢ + ‖hᵢ‖ · vᵢ / ‖vᵢ‖

Addition rather than replacement, because replacement made the activation norm explode (~20× at
layer 0, ~100,000× at layer 1). The subject layer ℓ is passed **as text**, so one oracle serves all
subject layers.

#### 2.0.1 The only two things that vary

| Variable | Levels |
| --- | --- |
| **Which backbone carries the LoRA** | Qwen3-8B, Llama-3.1-8B, Mistral-7B |
| **Which model supplies the injected activations** | 9 sources: {Qwen3-8B, Llama-3.1-8B, Mistral-7B} × {base M₀, leaf-Taboo, wave-Taboo} |

3 backbones × 9 activation sources would be 27 combinations, which is exactly the grid — each cell
of the 3 × 3 backbone grid contributes its 3 activation sources (base, leaf, wave) to that cell's
oracle backbone.

#### 2.0.2 base AO vs FT-AO — the actual manipulation

This is the single most important detail and the easiest to get wrong.

- A **base AO** is trained on activations from the **unmodified** M₀. This is what Karvonen et al.
  always do — *"always from the base model before LoRA fine-tuning."*
- An **FT-AO** is trained on activations from a **Taboo-fine-tuned subject**. This is our
  deviation from their recipe, and it is precisely the manipulation that produced anti-reading in
  the prior paper.

The exposure works like this: the generic AO corpus text is fed through the *Taboo subject* to
produce the activations. The corpus never mentions the secret word. But the subject represents it
persistently across a million generic examples, and the oracle trains against that signal — and
learns not to verbalise it. That is why the effect is interesting rather than trivial.

#### 2.0.3 What changes structurally in the cross-model setting

In Karvonen's setup the subject and oracle are literally the same weights: activations are produced
by *temporarily disabling the LoRA adapter* mid-forward-pass. Cross-model, that trick is impossible
— the subject is a separate model with separate weights. Two consequences:

**Both models must be resident during training.** Two 8B models in bf16 ≈ 32 GB, which fits
comfortably on an 80 GB A100 or H100. **Generate activations on the fly; do not pre-cache them.**
Caching the full corpus would be roughly 80 GB per subject per layer — about 2 TB across the nine
activation sources at three layers — which would dominate the storage budget for no benefit.

**Subject layers are chosen by relative depth**, matching the recipe's own 25/50/75% convention:

| Subject backbone | Layers | 25% | 50% | 75% |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-8B | 36 | L9 | **L18** | L27 |
| Llama-3.1-8B | 32 | L8 | **L16** | L24 |
| Mistral-7B | 32 | L8 | **L16** | L24 |

Evaluation uses the 50% layer, as in the original.

**Placeholder count follows the subject's tokenizer.** The same corpus text tokenises to different
lengths under Qwen (151,936 vocab), Llama (128,256) and Mistral (32,768), so K varies by subject
family. This is handled automatically — emit K placeholders for K vectors — but it does mean
oracles in different cells see slightly different sequence-length distributions. Measured fertility
ratios are within 6% of each other, so the effect should be small; worth reporting, not worth
correcting for.

**Expect modest extra cost per run.** Karvonen's ~10 H100-hours already includes a subject forward
pass, so the second model adds memory rather than a whole new computation. Budget 10–20% overhead
for the separate forward path rather than a doubling.

---

### 2.1 Grid of oracles

### Arm A — the kinship grid (the headline experiment)

**Both sides vary.** Subject backbone and oracle backbone are crossed, giving a full 3 × 3 grid of
backbone *pairs*. All three backbones are d = 4096, so **no dimension adaptation is used anywhere
in this arm** — raw vectors are injected directly, and the only thing that differs between cells is
which two models are paired.

| Backbone | `d` | Layers | Vocab | Used as subject | Used as oracle |
| --- | ---: | ---: | ---: | :---: | :---: |
| Qwen3-8B | 4096 | 36 | 151,936 | ✅ | ✅ |
| Llama-3.1-8B-Instruct | 4096 | 32 | 128,256 | ✅ | ✅ |
| Mistral-7B-Instruct-v0.3 | 4096 | 32 | 32,768 | ✅ | ✅ |

#### Why the subject side must vary too

An earlier draft of this plan held the subject fixed at Qwen3-8B and varied only the oracle. That
was wrong, for three reasons — the third is decisive:

1. **Generality.** A result measured only on Qwen subjects invites the obvious objection that it is
   a Qwen artifact. The prior paper already lists single-backbone as its main limitation; repeating
   that on the subject side would undercut the contribution.
2. **Direction.** Qwen-oracle-reading-Llama and Llama-oracle-reading-Qwen are different cells at the
   *same* kinship (Jaccard 0.643). If the kinship hypothesis is right they should behave alike.
   That is a strong, nearly free test — and it is impossible without subject-side variation.
3. **Statistical power.** The headline analysis regresses suppression magnitude on kinship. Varying
   only the oracle gives **three points**. A three-point regression supports no claim at all. The
   3 × 3 grid gives nine cells spanning four distinct kinship levels, each with replication.

#### The kinship levels the grid covers

Measured, not assumed — from `experiments/results/tokenizer_kinship.json`:

| Backbone pair | Tokenizer Jaccard | Cells at this level |
| --- | ---: | --- |
| self-pairs (Qwen↔Qwen, Llama↔Llama, Mistral↔Mistral) | **1.000** | 3 |
| Qwen ↔ Llama | **0.643** | 2 (both directions) |
| Llama ↔ Mistral | **0.071** | 2 (both directions) |
| Qwen ↔ Mistral | **0.061** | 2 (both directions) |

Four levels, nine cells, with two-directional replication at every cross level. That is a
regression worth running, and the paired directions double as the symmetry test.

#### Oracles required

To measure suppression in one cell you need a diagonal (own) and at least one off-diagonal (cross)
concept, plus a base AO as the denominator. So **2 concepts per cell** — leaf and wave, the two
largest effects in the prior paper's tables:

> per cell: 1 base AO + 2 concept FT-AOs = **3 oracles**
> × 9 cells = **27 oracles**

Each oracle is defined by the pair *(oracle backbone, activation source)*. Base AOs cannot be
shared across cells: a Llama oracle trained on Qwen base activations is a different model from a
Llama oracle trained on Llama base activations, and both are needed as their cells' controls.

#### Per-cell evaluation

|  | test subject: leaf | test subject: wave |
| --- | --- | --- |
| base AO (control) | cross | cross |
| FT-AO trained on **leaf** | **own** | cross |
| FT-AO trained on **wave** | cross | **own** |

Nine such grids × 5 capture regimes.

**Prediction.** Diagonal suppression is strongest in the three self-pair cells, intermediate at
Jaccard 0.643, weakest at 0.061–0.071 — and the two directions at each cross level agree with each
other.

### Arm B — multi-subject oracles (optional)

A separate question the grid does not answer: what happens to an oracle trained on activations from
*several* Taboo subjects at once? The prior paper covered multi-concept **subjects** (leaf+moon) but
never multi-subject **oracles**.

Note this experiment removes the own/cross contrast — if the oracle saw every concept in training,
every test cell is "own." It is therefore a complement to Arm A, not a substitute: it asks whether
the blind spot generalises to all exposed concepts or dilutes across them. Budget 2 oracles
(one Qwen-backbone, one Mistral-backbone, each trained on a mixture of all five Qwen Taboo
subjects) if funds allow.

### Arm C — the adaptation ladder

Arm C keeps the **subject fixed at Qwen3-8B** and uses oracle backbones that bracket its
`d_model = 4096` from both sides. Every backbone is in the Qwen3 family, so **tokenizer kinship is
pinned at 1.000 and only the shape-reconciling operation varies.**

| Oracle backbone | `d` | Layers | vs subject's 4096 | Operation | Information |
| --- | ---: | ---: | --- | --- | --- |
| **Qwen3-4B** | 2560 | 36 | smaller | **truncate** 4096 → 2560 | lossy, irrecoverable |
| Qwen3-8B *(existing)* | 4096 | 36 | equal | **none** | lossless — the reference |
| **Qwen3-14B** | 5120 | 40 | larger | **zero-pad** 4096 → 5120 | lossless |

Dimensions verified from HuggingFace `config.json`. Qwen3-32B is also d = 5120 but has 64 layers
and costs far more to train, so Qwen3-14B is the better "larger" pick.

A fourth rung, at no extra training cost: run the **fixed random projection** (Johnson–Lindenstrauss)
as an alternative to truncation at evaluation time on the Qwen3-4B oracle, to test whether the
principled downscale beats naive coordinate-dropping.

**Oracles needed: 6** — for each of Qwen3-4B and Qwen3-14B, one base AO plus leaf and wave FT-AOs,
all trained on Qwen3-8B activations. The middle rung is the existing Qwen3-8B self-cell, reused.

#### Predictions, and why the size confound does not bite

The obvious objection is that 4B, 8B and 14B differ in *capability*, not just in the operation. Two
things handle it:

1. **Suppression is measured within an oracle**, as own-concept performance against cross-concept
   performance for that same model. A uniformly stronger oracle raises both and does not
   mechanically reduce the gap.
2. **The two accounts make different predictions**, so they are distinguishable:

| Account | Qwen3-4B (truncate) | Qwen3-8B (no-op) | Qwen3-14B (pad) |
| --- | --- | --- | --- |
| *It is really about model capability* | worst | middle | best — **monotone in size** |
| *It is really about the operation* | worst | fine | fine — **pad ≈ no-op, both ≫ truncate** |

If reading quality degrades monotonically with size, the ladder is measuring capability. If 8B and
14B are comparable while 4B falls off, the ladder is measuring information loss from truncation.
Bracketing the subject on both sides is what makes this test possible — a ladder built only from
smaller models could not separate the two.

---

## 3. Total training load

### What reuse actually covers

Reusing the prior paper's Qwen3-8B oracles removes **one cell of nine** — the Qwen←Qwen self-cell —
worth 3 oracles. It does **not** cover the rest of the Qwen row: a Qwen-backbone oracle trained on
*Llama* or *Mistral* activations has never been trained, because the prior paper only ever fed
Qwen activations to Qwen oracles.

That matters because those two cells carry the direction-symmetry test. Qwen-oracle-reading-Llama
and Llama-oracle-reading-Qwen sit at identical kinship (0.643); if they behave alike, the kinship
account is strongly supported. Dropping the Qwen row entirely saves $150 and loses that test.

| Option | Cells covered | New oracles | Grid cost | Keeps symmetry test? |
| --- | ---: | ---: | ---: | :---: |
| Llama + Mistral backbones only | 7 of 9 | 18 | $450 | ❌ |
| **+ the two remaining Qwen cells** | **9 of 9** | **24** | **$600** | ✅ |

Recommendation: pay the $150. It is the cheapest strong evidence in the whole project.

### Totals

| Item | Count | Cost each | Total |
| --- | ---: | ---: | ---: |
| New Taboo subjects | 6 | ~$4 | **~$25** |
| Arm A — Qwen←Qwen cell | 3 | $0 | **reused** |
| Arm A — Llama backbone × 3 subject sources | 9 | ~$25 | **~$225** |
| Arm A — Mistral backbone × 3 subject sources | 9 | ~$25 | **~$225** |
| Arm A — remaining Qwen cells (symmetry test) | 6 | ~$25 | **~$150** |
| **Arm C — Qwen3-4B oracles (truncate)** | **3** | ~$13 | **~$39** |
| **Arm C — Qwen3-14B oracles (pad)** | **3** | ~$42 | **~$126** |
| Arm B (multi-subject, optional) | 2 | ~$25 | ~$50 |

**Oracle training subtotal: ~$765** for 33 trained oracles across 30 runs, or **~$615** without the
symmetry cells.

Arm C unit costs scale with backbone size: Qwen3-4B is roughly half an 8B run, Qwen3-14B roughly
1.75×. Memory is fine in both cases — Qwen3-14B in bf16 (28 GB) plus the resident Qwen3-8B subject
(16 GB) is ~44 GB, comfortable on an 80 GB card.

### This changes the funding tiers

Crossing both sides costs more than the earlier one-sided plan, so the tiers are redrawn around
**how much of the grid** is bought rather than around extras:

| Tier | Scope | New oracles | What it supports |
| --- | --- | ---: | --- |
| **~$1,000** | Arm A, Llama + Mistral backbones only (7 of 9 cells), no Arm C | 18 | Does a foreign oracle avoid the blind spot? A comparison across three kinship levels, no symmetry test, no adaptation ladder. |
| **~$1,800** | **Full 3 × 3 grid + the Qwen3-4B / Qwen3-14B adaptation ladder** | 30 | The kinship regression at four levels, the direction-symmetry test, *and* the pad-vs-truncate result with the capability confound controlled. |

Rough upper-tier total: $765 oracles + $25 subjects + ~$180 capture + ~$120 evaluation + ~$170
analysis + ~$120 storage + ~$60 judging + ~$50 pilots ≈ **$1,490**, plus 20% contingency ≈
**$1,790**. Contingency is trimmed from 30% because reusing the Qwen self-cell removes the runs
most likely to need repeating — that recipe is already known to work.

**The trade-off to be explicit about:** at $2,000 you can afford the full grid *or* triple-seed
error bars, not both. Recommendation is the grid. The two-directional replication at each cross
kinship level provides a partial substitute for seed replication, and "you only tested one model
family" is a more damaging criticism than "you ran each condition once" — the prior paper already
carries the second one.

---

## 4. Free pilot on local hardware (RTX 2080 Ti, no grant money)

A miniature of Arm A that runs on an 11 GB desktop card, using **already-released** oracle
checkpoints so nothing needs training.

| Model | `d` | Layers | Vocab | Released AO checkpoint? |
| --- | ---: | ---: | ---: | --- |
| Qwen3-1.7B | 2048 | 28 | 151,936 | ✅ yes |
| Llama-3.2-1B-Instruct | 2048 | 16 | 128,256 | ✅ yes |

Both are d = 2048, different families, different tokenizers, different depths — the same structure
as Arm A, three sizes down. Feed raw Qwen3-1.7B activations to the off-the-shelf Llama-3.2-1B
oracle and see whether it reads anything. Both fit in VRAM simultaneously in fp16 (~6 GB).

⚠️ **Asymmetric evidence.** A positive result here strongly supports Arm A. A negative result is
nearly uninformative — cross-family representational alignment is known to collapse below ~2B
(CKA ≈ 0.2 at 1–3B versus ≈ 0.6 within family). Do not let a null at 1.7B cancel the 8B experiment.

⚠️ **Turing has no bf16.** Compute capability 7.5 means fp16 or fp32 only, and no FlashAttention-2.
For activation capture specifically, prefer fp32 at this scale — fp16 risks overflow on exactly the
massive-activation dimensions the analysis cares about. A 1.7B model in fp32 is ~6.8 GB and fits.

---

## 5. Backbones deliberately excluded, and why

| Model | Why not |
| --- | --- |
| Gemma-2-9B-IT (d = 3584, verified) | Tokenizer Jaccard 0.066 vs Qwen3 — nearly identical to Mistral's 0.061 — but the dimension mismatch means it needs adaptation, whereas Mistral does not. Mistral gives the same kinship point for free. Keep Gemma as a spare, or use it in Arm C. |
| Qwen3-14B / Qwen3-32B | Same family as the anchor, so they add no kinship variation, and they cost more per run. |
| Llama-3.3-70B | Needs multi-GPU and large-memory cards (Karvonen used 4×H200 for this one). A scale check for a future grant, not this one. |
| Yi-1.5-34B | Same reasoning; also a heavier download and slower activation capture. |

*(Dimensions for the excluded models are not listed because none of them drive a decision here; only Gemma's has been verified.)*

---

## 6. Open decisions to settle before booking compute

1. **Reuse or retrain the Qwen3-8B anchor oracles?** Several already exist from the prior paper.
   Reuse only if the injection recipe is byte-identical to the new runs — which depends on resolving
   the Appendix B.3 discrepancy (layer 18 + λ vs layer 1 + norm-matching). The budget above assumes
   retraining for consistency; reusing saves ~$75.

2. **Original recipe or the improved one?** *Building Better Activation Oracles* (arXiv:2606.02609)
   reports a substantially better recipe — a new conversational dataset built from chain-of-thought
   suffixes, feeding 5 contiguous layers (21–25) instead of one, on-policy rollouts, and stronger
   injection — lifting chance-adjusted AObench from +0.244 to +0.435 and cutting hallucination.
   Using it yields better oracles but breaks direct comparability with the prior paper's numbers;
   using the original keeps the 94% → 35% baseline meaningful. **Recommendation: original recipe for
   the grid** (the claim is a comparison between cells, so absolute oracle quality matters less than
   consistency), with a single improved-recipe oracle trained as a sanity check that the effect is
   not an artifact of a weak recipe.
2. **Layer-label convention for foreign donors.** Qwen3-8B L18 = 50% depth; Llama-3.1-8B 50% = L16;
   Mistral-7B 50% = L16. Decide whether the oracle prompt says the absolute index or the relative
   depth, and keep it fixed across all arms.
3. **Which subject layer to capture.** The prior work used L18 of Qwen3-8B. Held at 50% relative
   depth, the Llama and Mistral equivalents are L16. Confirm with the CKA sweep before committing.

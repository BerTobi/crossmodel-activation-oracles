# Minimal experiment — $500

The smallest spend that produces a publishable result, staged so each gate can kill the project
before the next tranche is spent.

**Target:** three kinship points, a full own-vs-cross matrix at each, plus mechanism and controls.
**Estimated cost: $416**, leaving ~$84 unallocated.

---

## Settle this before spending anything

> ⚠️ **Match the prior paper's recipe, do not switch to Karvonen's.**
>
> The entire $500 design depends on reusing the existing Qwen3-8B oracles as the kinship-1.000
> anchor. That reuse is only valid if the new oracles are trained identically. Our Appendix B.3
> describes injection as additive at **AO layer 18** with `λ = 1.0`; Karvonen's published recipe is
> norm-matched addition at **AO layer 1**. These are different models.
>
> **Whichever our existing oracles actually used, use that.** Switching recipes mid-project would
> confound the anchor cell with the new cells and destroy the comparison — for zero benefit, since
> the claim is a contrast between cells, not an absolute quality number.
>
> One consequence to decide with it: injecting at "layer 18" means 50% depth in a 36-layer Qwen but
> 56% in a 32-layer Llama or Mistral. **Use relative depth** so the foreign oracles are matched to
> the anchor.
>
> **A cost to name, not hide:** if our recipe injects at ~50% depth rather than Karvonen's layer 1,
> the foreign oracle has only half its layers in which to learn the basis rotation that the
> self-oracle never needed. This makes cross-model reading harder than it would be under the
> published recipe. It is still the right call — comparability with the anchor is the whole
> experiment — but it means a Gate 1 failure is partly attributable to the recipe, and the
> write-up should say so.

---

## What this experiment actually decides

This is **not** a confirmation of a prediction. It discriminates between two mechanisms that fit the
prior paper equally well, and **both outcomes are publishable**.

| Story | Where anti-reading lives | Predicts |
| --- | --- | --- |
| **Input-side** | In the *subject's activations*: they carry both "leaf" and "don't say leaf," and any oracle trained on them long enough learns to honour the second signal. Backbone irrelevant. | **No gradient.** Foreign oracles suppress as much as the self-oracle. |
| **Weight-sharing** | In the *oracle's shared base*: a LoRA on the same weights already has the circuits the Taboo fine-tune targeted, so the "don't emit leaf" policy is cheap to install. A foreign backbone lacks them. | **A gradient.** Suppression falls as oracle and subject diverge. |

The prior paper's concept-specificity result — the blind spot follows the AO's training concept
even on multi-concept subjects — is consistent with *both*. This experiment separates them.

**If there is no gradient, that is the more alarming result for auditing, not a failure:** no choice
of auditor backbone protects you, and the only remaining defence is an oracle never trained on the
auditee's activations at all (the zero-shot / universal-oracle direction).

### Pre-registered dependent variable

Foreign oracles are expected to read *worse on average* (Li et al.; UAV), so their base AOs may
land at 55% rather than 94%. An absolute-drop measure would then be uninterpretable. **Decide the
DV now:**

- **Primary:** the prior paper's normalised readout measure — log rank-ratio
  Δℓ = log₁₀(rank_own / rank_cross), or equivalently own/cross ratio of P(c⋆) — computed within
  each oracle backbone. Ratio, not difference.
- **Secondary:** exact and semantic recovery, reported alongside for legibility.
- **Contrast:** own FT-AO vs **cross** FT-AO within the same backbone. The base AO is the
  denominator for feasibility, not the primary comparator — the prior paper's 94 → 35 headline is
  really 71 → 35 once general FT degradation is separated out.

### Immune or blind? The internal probe decides

"Less suppression in the foreign oracle" has two readings, and only one is interesting:

| Reading | Concept decodable *inside* the foreign FT-AO? | Verbalised? |
| --- | :---: | :---: |
| **Immune** — the claim | yes | yes |
| **Blind** — a floor effect | no | no |

A linear probe on the foreign FT-AO's own hidden states (the prior paper's L33 analysis) is the
test. **Run it the moment Gate 2 data exists.** It is the interpretation of the headline, not a
Stage 3 extra.

---

## Design

Subject side is fixed: the **existing Qwen3-8B Taboo subjects**, cooperative α = 1.0, concepts
**leaf / moon / wave**. No new subjects are trained. Only the oracle backbone varies.

*(EXPERIMENTS.md uses two concepts because it trains new subjects; here the third concept is free
because all five Qwen subjects already exist. The extra oracle per backbone is $25 well spent.)*

| Cell | Oracle backbone | `d` | Tokenizer Jaccard vs Qwen3 | Oracles | Cost |
| --- | --- | ---: | ---: | ---: | ---: |
| Anchor | Qwen3-8B | 4096 | **1.000** | 4 | **reused — $0** |
| Mid | Llama-3.1-8B-Instruct | 4096 | **0.643** | 4 | $100 |
| Far | Mistral-7B-Instruct-v0.3 | 4096 | **0.061** | 4 | $100 |

All three at d = 4096, so **no padding, truncation or projection anywhere**. The only thing that
changes between cells is the backbone. Four oracles per cell: one base AO plus leaf, moon and wave
FT-AOs, giving a 3 × 3 own/cross matrix per backbone.

**What is deliberately cut from the full plan:** the subject side stays fixed (no Llama or Mistral
Taboo subjects), so there is no direction-symmetry test; and the pad/truncate ladder
(Qwen3-4B / Qwen3-14B) is dropped entirely. Both are the first things to add when more money
arrives.

**Three kinship points is the minimum that can show a monotone ordering.** Two points would only
license "different is better," which is far easier to explain away. Three is *not* enough for a
fitted regression with a confidence interval — do not describe it as one. The regression framing
in `PROPOSAL.md` belongs to the full 3 × 3 grid.

**Tokenizer Jaccard is a proxy, not a mechanism.** The oracle never tokenises the subject's text —
activations enter at placeholder positions — so vocabulary overlap has no direct effect on
injection. It matters only because it tracks representational similarity across families
(arXiv:2603.18908). **CKA between residual streams at matched depth is the direct covariate**; it
comes out of Stage 1 for free. Report both, lead with CKA.

---

## Stages and gates

### Stage 0 — free, on the RTX 2080 Ti

| Task | Status |
| --- | --- |
| Tokenizer kinship matrix | ✅ done — `experiments/results/tokenizer_kinship.json` |
| Confirm the existing Qwen oracles still load and reproduce the prior paper's numbers | to do |
| Small-scale pilot on Qwen3-1.7B ↔ Llama-3.2-1B (both d = 2048) — see below | to do |

**The pilot must not be the raw zero-shot version.** An earlier draft proposed feeding *raw*
Qwen3-1.7B activations to the released Llama-3.2-1B oracle with no training. That oracle has only
ever seen Llama activations; handing it vectors in an unrelated basis is a far harder task than
anything Stage 1–2 does, since those *train* on the foreign activations. A null would be
near-certain and would say nothing about the plan. Two versions are actually informative:

| Pilot version | What it tests | Cost |
| --- | --- | --- |
| **Procrustes-mapped zero-shot.** Fit an orthogonal map on shared inputs between the two 1.7B/1B models, map the Qwen activations into the Llama basis, then inject into the released Llama oracle. | Whether the information is *geometrically reachable* by a foreign oracle once the basis is aligned. | hours |
| **Reduced-data mini-training.** Train a Llama-3.2-1B oracle on Qwen3-1.7B activations with a cut-down corpus (~50k examples). | A genuine miniature of Stage 1. | a day or two on the 2080 Ti |

Both fit in 11 GB. Prefer the second if time allows; it is the only pilot that rehearses the actual
training path.

The evidence is still **asymmetric**: a positive result supports the 8B experiment, a negative one
is weak evidence because cross-family alignment is known to collapse below ~2B. Do not let a null
at 1.7B cancel the plan.

Turing has no bf16 — use fp32 for activation capture at this scale to avoid overflow on the
massive-activation dimensions.

### Stage 1 — feasibility gate · $80

| Item | Cost |
| --- | ---: |
| Rent a 4090 for 8B activation capture, CKA and Procrustes between Qwen3-8B ↔ Llama and ↔ Mistral | $20 |
| Train **2 foreign base AOs** (Llama and Mistral, both on Qwen3-8B **base** activations) | $50 |
| Evaluate both on the existing Taboo subjects, 5 regimes | $10 |

Base AOs are needed as controls regardless, so they double as the feasibility test at no waste.

> **GATE 1.** Do the foreign base AOs recover hidden concepts from Qwen Taboo subjects?
> Roughly **≥60% exact recovery** is needed for a 94 → 35 style collapse to be measurable against.
> If Mistral fails but Llama passes, **do not collapse to two points** — swap in another d = 4096
> backbone (candidates exist; verify `hidden_size` on HuggingFace before committing, as was done
> for the models here). If both fail, stop — and that is still a reportable finding about the
> limits of raw cross-model injection, with the depth-of-injection caveat above attached.

### Stage 2 — the headline · $165

| Item | Cost |
| --- | ---: |
| Train **6 FT-AOs** — leaf, moon, wave × Llama, Mistral, all on Qwen Taboo activations | $150 |
| Evaluate the 3 × 3 own/cross matrix per backbone, 5 regimes | $15 |

> **GATE 2.** Does own-vs-cross suppression (the pre-registered Δℓ) weaken as kinship falls?
> Weight-sharing story: strong at Qwen (replicating the prior result), weaker at Llama, weakest at
> Mistral. Input-side story: roughly equal at all three.
> **Then immediately run the internal probe on the foreign FT-AOs** to establish immune vs blind
> before drawing either conclusion.

### Stage 3 — mechanism and controls · $45

| Item | Cost |
| --- | ---: |
| AO-internal linear probes + LogitLens across oracle layers | $15 |
| Layer-range LoRA ablations on the foreign oracles | $20 |
| Controls: shuffled activations, wrong-subject activations, zero-ablated injection, text-only baseline | $10 |

The controls are not optional. arXiv:2509.13316 shows verbalizers often report priors rather than
activation content, and a cross-model oracle is more exposed to that, not less. A reviewer will ask.

### Running costs

| Item | Cost |
| --- | ---: |
| Storage, ~1 month | $15 |
| Claude judging (Sonnet, batch API) | $15 |

---

## Budget

| Stage | Cost |
| --- | ---: |
| Stage 0 — desktop | **$0** |
| Stage 1 — feasibility gate | $80 |
| Stage 2 — headline experiment | $165 |
| Stage 3 — mechanism and controls | $45 |
| Storage and judging | $30 |
| **Subtotal** | **$320** |
| Contingency (30%) | $96 |
| **Total** | **$416** |
| Unallocated | $84 |

Keep the $84 genuinely unallocated. The evaluation and analysis figures are derived from the
experiment's structure rather than measured, so they are the least reliable numbers here — and
Stage 2 is the first time anyone has trained an oracle this way, so at least one run will probably
need repeating.

---

## What $500 does and does not buy

**Delivers:** a three-point kinship gradient, a full own-vs-cross matrix at each point,
readout-side mechanistic evidence carried over from the prior paper's toolkit, and the controls
needed to show the oracle is reading activations rather than reciting priors. That is a paper.

**Does not deliver:**

| Missing | Cost to add later | Why it is in this order |
| --- | ---: | --- |
| **Qwen3-14B oracles (4)** | **~$170** | **First.** Same family, same tokenizer, *different weights*, lossless pad. The three current points can separate input-side from weight-sharing, but not two versions of weight-sharing: "requires literally identical base weights" vs "scales with shared geometry." Qwen3-14B splits them — identical-weights predicts it is immune; shared-geometry predicts it is suppressed like Qwen3-8B. Most mechanistic value per dollar of anything on this list. |
| Direction-symmetry test (Llama/Mistral Taboo subjects + the Qwen row) | ~$225 | Removes the "Qwen-subject artifact" objection and tests whether the effect depends on the pair or the direction. |
| Repeat seeds / error bars | ~$150 | Reviewers will ask; the two-direction replication above partly substitutes. |
| Qwen3-4B (truncate rung) | ~$40 | Completes the pad/truncate ladder. Cheap, but lower priority than the above. |
| Multi-subject oracles | ~$50 | A complement to the grid, not a substitute. |

**The honest caveat for the write-up:** with the subject side fixed at Qwen3-8B, the result reads
"for Qwen subjects, foreign oracles are more reliable." Generality across subject families is the
single biggest gap, and it should be stated as a limitation rather than glossed. It is also the
first thing the next tranche should buy.

---

## Order of operations and timeline

| Step | What | Calendar time |
| --- | --- | --- |
| 1 | Resolve the injection-recipe question; fix the layer convention; fix the DV. | days |
| 2 | Stage 0 on the desktop. Cost: nothing. | ~1 week, overlapping with 1 |
| 3 | Stage 1. Stop at Gate 1 before releasing Stage 2 funds. | ~1 week — the first cross-model run will need debugging |
| 4 | Stage 2, then Gate 2, then the internal probe *immediately*. | ~1 week (6 runs, parallelisable across instances) |
| 5 | Stage 3 remainder only if Gate 2 shows something worth explaining. | ~3–4 days |

**Realistic total: 3–4 weeks.** Most of the slack is in step 3. GPU-hours are ~150 across all stages,
which is a few days sequential or a day or two spread across rented instances; calendar time is
dominated by getting the first foreign oracle to train correctly, not by compute.

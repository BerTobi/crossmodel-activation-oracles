# 3-day sprint — MATS application (Neel Nanda stream)

**Goal:** one or two genuinely new results, written up honestly, in 72 hours.
**Constraint:** nothing that needs an 18-hour training run to pay off.
**Compute:** one rented A100 80GB for ~20 hours (~$30) + the desktop. Well under $500.

**Why this fits the stream:** the prior paper builds on Cywiński et al. (Taboo organisms — Neel's
group) and Karvonen et al. (AOs — Neel co-authored the follow-up, *Building Better AOs*). Say so.

---

## Priority 1 — The mechanism question, on models you already have · Day 1

**No training. Uses the existing Qwen3-8B FT-AOs, base AO, and Taboo subjects.**

The prior paper shows the leaf-FT-AO fails to say "leaf" when reading the leaf-subject. The open
question — and the one a mech-interp reviewer asks first — is **what triggers the suppression**:

| Hypothesis | Trigger | What a foreign oracle would do |
| --- | --- | --- |
| **Concept-keyed** | The concept representation itself, from any source. The oracle has learned "never say leaf." | Ambiguous — depends whether learning the policy needs shared circuits |
| **Source-keyed** | A signature specific to Taboo-subject activations ("I am hiding something"). | **Not immune** — the signature is in the input, any oracle can learn to detect it |

Two inference-only tests separate them:

### Test 1 — concept from a non-Taboo source

Feed the **leaf-FT-AO** activations from the **base** Qwen3-8B (never Taboo-trained) on text where
"leaf" is salient — e.g. a system prompt *"Your secret word is leaf. Give the user a hint,"* or plain
prose about autumn leaves. Ask: *What is the secret word / main concept in this text?*

- Reads "leaf" fine → **source-keyed**. Suppression is triggered by the Taboo signature, not the
  concept. Cross-model immunity is unlikely.
- Suppresses "leaf" → **concept-keyed**. The FT-AO is broken for that concept *everywhere*, not
  just on its training subject. New, and striking on its own.

Controls: base-AO and moon-FT-AO on the same activations (both should read "leaf"). Every existing
regime avoided this cell — OFFTOPIC has no concept present, and all other regimes use Taboo
subjects — so it is genuinely untested.

### Test 2 — steering dose-response

The prior paper already computed δ = mean(leaf-subject activations) − mean(base activations) for
Δ-LogitLens. Reuse it. Take base-subject activations on a neutral prompt, add **α · δ** for
α ∈ {0, 0.25, 0.5, 1, 2, 4}, and feed to base-AO, leaf-FT-AO, moon-FT-AO.

- base-AO should read "leaf" increasingly with α — confirms δ carries the concept.
- leaf-FT-AO suppresses at **all** α → concept-keyed.
- leaf-FT-AO reads at low α, suppresses at high α → the Taboo signature rides in the same
  direction, dose-dependently. Also interesting.
- moon-FT-AO tracks the base-AO → control.

One figure: P(leaf) vs α, three lines. This is the kind of plot that lands in an application.

**Cost:** both 8B models in bf16 fit on one A100. A few hours. ~$8.

---

## Priority 2 — Cross-model geometry at 8B · Day 1–2, in parallel

**No training.** On the same A100: run Qwen3-8B, Llama-3.1-8B, Mistral-7B over the Taboo prompt
set (HINT, THINK, SAMETEXT), capture residual streams at 25 / 50 / 75% depth, compute **CKA** across
every layer pair and fit **Procrustes** maps on SAMETEXT pairs.

Output: a CKA heatmap per model pair + Procrustes cosine per layer. Answers "is cross-model
activation reading geometrically plausible on *this* distribution?" — a feasibility figure the grant
needs and nobody has for this setting. Pairs with the tokenizer-kinship numbers already measured
(Qwen↔Llama 0.643, Qwen↔Mistral 0.061).

**Cost:** ~4 hours. ~$6.

---

## Priority 3 — First cross-model AO on concealed information · Day 2

**The only training in the sprint, de-risked by cutting the corpus.**

Karvonen's full recipe is ~1M examples, ~10 H100-hours. Cut it to **~100k examples (~1 hour on
H100, ~2 on A100)** and train a **Llama-3.1-8B oracle on Qwen3-8B base activations**. Quality will
be lower — that is fine for a yes/no pilot, and it means a failed attempt costs an hour, not a day.
Evaluate on the existing leaf/moon/wave Taboo subjects, HINT regime, exact + semantic recovery +
P(c⋆).

- Recovers anything above the shuffled-activation control → **first demonstration of a
  cross-family activation oracle reading concealed information**, even at reduced quality.
- Recovers nothing → a clean negative that pairs with the CKA result to say *why*.

**Engineering risk is real:** Karvonen's code produces activations by toggling the LoRA on the
same model. Loading a separate subject model is a code change. Budget Day 1 evening for it; if it
is not training by Day 2 noon, drop Priority 3 and put the time into the write-up.

**Cost:** 2–3 attempts × ~$5. Under $20.

---

## Priority 4 — Desktop pilot · Day 1, background, free

Procrustes-mapped Qwen3-1.7B → released Llama-3.2-1B oracle (both d = 2048). Fits on the 2080 Ti
in fp32. Weak evidence either way, but if it reads anything it is a nice teaser and costs nothing.
Run it while the A100 does Priority 1.

---

## Explicitly NOT doing

- Any full-recipe 18-hour oracle run
- The 3 × 3 grid, Qwen3-14B, or anything from the $500 plan beyond its Stage 0
- Resolving the Appendix B.3 recipe question by retraining — just **state** what the existing
  oracles used and match it in Priority 3

---

## Schedule

| Day | Morning | Afternoon | Evening |
| --- | --- | --- | --- |
| **1** | Rent A100. Test 1 + Test 2 code and first runs. Desktop: Priority 4. | Priority 2 capture running; analyse Test 1/2. | Engineer the two-model activation path for Priority 3. |
| **2** | Launch reduced-corpus foreign AO. Finish CKA/Procrustes figures. | Evaluate foreign AO. One retry if needed. | **Stop experiments.** Start writing. |
| **3** | Write-up: figures, claims, limitations. | Polish. Submit with margin. | — |

**Hard rule:** no new experiments after Day 2 evening. A half-finished third result is worth less
than a clean write-up of two.

---

## The write-up — what it should say

1. **One paragraph** on the prior paper: FT-AOs become concept-specific anti-readers (94 → 35),
   readout-side mechanism, arXiv link.
2. **The mechanism result** (Priority 1): concept-keyed or source-keyed, with the dose-response
   figure. Say what it implies for cross-model oracles either way.
3. **Feasibility** (Priority 2): CKA/Procrustes figure. One sentence on what it does and does not
   license.
4. **The cross-model pilot** (Priority 3) if it ran — reported at face value, including the reduced
   corpus and whatever recovery rate it hit, or the clean negative.
5. **What I'd do in the stream:** the 3 × 3 kinship grid, the input-vs-weight discrimination at
   scale, and the pad/truncate ladder. Tie to *Building Better AOs* explicitly.
6. **Limitations, unhedged.** Single seed, single subject family, reduced corpus in the pilot.

Calibrated claims and a visible prioritisation trail are what this stream selects for. The
prioritisation *is* part of the result.


---

## ACTUAL SPRINT RUN (decided 2026-09-01, launched 2026-09-02)

After `prev paper data/` arrived, the run was re-scoped to Tobias's own organism:

| | |
| --- | --- |
| **Oracle** | `Qwen/Qwen3-14B` (d=5120, 40 layers) — LoRA r64/α128, Karvonen recipe |
| **Subject** | `Qwen/Qwen3-8B` @ `b968826d…` + **`Mpp_clock_adapter`** (the validated clock organism from the Misaligned-Oracles capstone), kept ACTIVE during capture |
| **Adaptation** | zero-pad 4096 → 5120 (lossless); injection norm-matched at oracle L1 (Karvonen default — the capstone handoff confirms this was the recipe, not "layer 18") |
| **Anchor** | released `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B` (C1) reading the same clock activations |
| **Eval** | `code/collect_readouts_xm.py` — cache-once shim across oracles of different width; `checker_clock.py`; `taboo_direct_test.txt` (denial) + `taboo_standard_test.txt` (hint) |
| **Data** | full Karvonen mixture incl. past-lens (needs lmsys login on the pod) |
| **Compute** | secure H100 SXM, pod `4p68a6o3q95u3q`, est. 10–15 h, $35–50 |

**What it tests.** Does an oracle whose *backbone is not the organism* — different weights, same family — still
suppress the organism's own secret when trained on its activations? The capstone's C2 had the organism AS its
backbone, so backbone and activation source were never separated.

**Caveat (stated up front).** Versus C1 this cell changes two things at once: oracle size/weights AND
contaminated training activations. The 8B-backbone version (C3-8B: clean Qwen3-8B oracle on clock activations)
is the single-variable disambiguator and is the first follow-up run.

**Sequence.** Smoke (AO_DEBUG=1, 20 batches) → full run detached with a persistent monitor → Day 2 eval with
`collect_readouts_xm.py` → write-up.

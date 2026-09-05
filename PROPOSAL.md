# Cross-Model Activation Oracles — Research Proposal (v0.3)

Prior work by this author: *When Activation Oracles Learn Not to Read: Concept-Specific Blind
Spots in Fine-Tuned Oracles* (arXiv:2607.23379).

---

## 1. The novelty claim as stated does not survive

> "Nobody has trained AOs from one model that can read activations from another model."

At least four published works do this. The Activation Oracles paper itself cites two of them under
a Related Work heading literally titled *"Explaining activations from a different model."*

| Work | What it did | Cross-family? |
| --- | --- | --- |
| **Patchscopes** — Ghandeharioun et al., ICML 2024 (arXiv:2401.06102) | Training-free hidden-state patching into a second model's forward pass, with optional learned affine map `f`. Vicuna-7B → Vicuna-13B inspector. | Same family only |
| **Meta-models** — Costarelli, Allen & Field, Oct 2024 (arXiv:2410.02472) | Phi-2 trained to answer NL questions about **Llama-3.1-8B** activations; also InternLM2.5-7B on the same donor. | Yes — cross-family, cross-tokenizer, cross-dimension |
| **Li, Guo, Huang, Steinhardt & Andreas**, Nov 2025 (arXiv:2511.08579) | Activation inserted as a continuous token at the embedding layer, **with a learned projection Π when dimensions differ**. Llama-3.1-8B ↔ Qwen3-8B bidirectional; Llama-3.1-70B on Llama-3.1-8B. | Yes |
| **Universal Activation Verbalizer (UAV)** — Zhao et al., May 2026 (arXiv:2605.25903) | Frozen shared Qwen3-4B decoder + per-donor MLP/Q-Former adapter → *n* soft tokens at the embedding layer. Donors: Llama-3.1-8B, Gemma-3 4B/12B, Yi-1.5-34B, Qwen3-4B. Includes a **4B decoder reading a 34B donor**. | Yes, "across model families and scales" |

Supporting machinery also exists: **Activation Space Interventions Can Be Transferred Between LLMs**
(arXiv:2503.04429) trains nonlinear autoencoder mappers between Llama 1B↔3B, Qwen↔Llama and
Gemma→Llama, and finds **tokenizer similarity strongly predicts cross-family transfer quality, with
nonlinear maps beating affine**. **vec2vec** (arXiv:2505.12540) translates between unrelated
encoders with no paired data. **Universal SAEs** (arXiv:2502.03714) build one concept space across
models.

**This is good news.** The dimension-bridging engineering is solved and citable, which frees the
project to make a scientific claim instead of an engineering one — and there is a much better claim
available, because the existing cross-model papers disagree with each other.

---

## 2. The contradiction in the literature

Two credible, well-executed results point opposite ways:

- **Li et al. (2511.08579) — strong privileged access.** Self-explanation decisively beats
  cross-model explanation. Their headline: *"a Llama-3.1-8B explainer is significantly better than
  Llama-3.1-70B, showing that matching the target model matters more than a 9× increase in
  parameter count."* Pre-aligning the projection helps (+15% on SAE explanations, 2.7× on real
  activations) but does not close the gap. Karvonen et al. cite exactly this to justify staying
  same-model.
- **UAV (2605.25903) — weak privileged access.** Self-decoding is stronger, but *"the performance
  gap remains moderate,"* and cross-model is competitive with LatentQA and AO self-explanation
  baselines.

The likely reconciliation, sharpened by the skeptical audit of this whole family
(Li, Ceballos Arroyo, Rogers, Saphra & Wallace, arXiv:2509.13316 — verbalizers *"often verbalize
their own knowledge rather than the information encoded in the target LLM's activation,"* and on
PersonaQA-Fantasy *"verbalization collapses to zero"*): **it depends on where the information
lives.** UAV's tasks are largely recoverable from reconstructable input text, which transfers
across models. Li et al.'s tasks require genuinely privileged access to internals, which does not.

### The regime the literature has never tested

Both camps evaluate information that the subject model is *not trying to hide*. This author's prior
paper covers the third regime, and finds the self-oracle catastrophically failing in it.

| Regime | Where the information lives | Best oracle | Evidence |
| --- | --- | --- | --- |
| **Input-recoverable** — classification, gist, sentiment | in the text | cross ≈ self | UAV |
| **Privileged, not suppressed** — PersonaQA, SAE features | in the weights | **self ≫ cross** | Li et al. |
| **Privileged *and* actively suppressed** — Taboo, backdoors, deception | in the weights, behaviorally concealed | **cross > self** ← *the open cell* | this project |

---

## 3. Thesis

> **Privileged access is a fair-weather advantage. A self-oracle reads its own internals better
> on average, but it inherits the subject's suppression along with its representations — so it
> fails precisely on information the subject was trained to conceal, which is the only information
> an auditor is hired to find. Cross-model oracles pay an average-fidelity tax in exchange for
> structural immunity to concept-specific anti-reading.**

The empirical backing already half-exists: the prior paper shows exact recovery of a hidden concept
collapsing from **94%** (base AO) to **35%** (own FT-AO) for cooperative α=1.0 subjects, while
cross-*concept* oracles hold at 71%. The target remains linearly decodable inside the oracle at
L33; suppression is localized to the **readout** (LogitLens degrades in late layers; ablating the
FT-AO LoRA at L18–23 restores target probability).

### Why cross-model should be immune (the mechanism)

Anti-reading lives in the readout — the interaction between the oracle's representation and **its
own unembedding geometry**. A same-backbone oracle shares the subject's tokenizer, vocabulary
geometry and base weights, so the "don't emit *leaf*" direction the Taboo fine-tune installs has a
natural home in the oracle. A foreign-backbone oracle shares none of it.

Central falsifiable prediction:

> **Anti-reading magnitude scales with representational kinship between oracle and subject.**

| Oracle ↔ subject | Predicted anti-reading |
| --- | --- |
| Identical backbone (Qwen3-8B ← Qwen3-8B) | Strongest — replicates prior result |
| Same family, different scale (Qwen3-4B ← Qwen3-8B) | Intermediate |
| Different family (Llama-3.1-8B ← Qwen3-8B) | Weak |
| Different family + distant tokenizer (Gemma-3 ← Qwen3-8B) | Weakest / absent |

**Make this quantitative, not ordinal.** The transfer literature has already identified the right
covariates, so the gradient can be pre-registered as a regression rather than a ranking:

- **Tokenizer overlap is the single best predictor of cross-model transfer.** Gorbett & Jana
  (arXiv:2603.18908) find exact-token-match correlates **r = 0.898** with stitched-generation
  success and vocabulary Jaccard **r = 0.822**. Gemma's Jaccard vs. other families is **< 0.07**,
  which is why Gemma is the natural "maximally distant" point.
- **CKA / SVCCA between residual streams** at matched relative depth gives a continuous
  representational-similarity axis (useful range CKA 0.5–0.88).
- Independent mechanism support: arXiv:2503.04429 transfers *steering* across families and finds
  tokenizer similarity gates quality. If suppression is a direction in the same geometry that
  steering exploits, transfer of *suppression* should obey the same law.

So the headline analysis becomes: **regress anti-reading magnitude on tokenizer Jaccard and CKA
across all oracle–subject pairs.** A significant positive slope is a far stronger result than a
four-row ordering, and it is registerable in advance.

---

## 4. Contributions

**C1 — Cross-model AOs on concealed information.** No cross-model verbalizer has been evaluated on
hidden-knowledge elicitation. Establish that a foreign oracle recovers Taboo concepts from
Qwen3-8B subjects at a useful rate. Go/no-go gate.

**C2 — The immunity result (headline).** Replicate the base / cross / own FT-AO grid with foreign
oracle backbones; test the kinship gradient. Carry the prior paper's mechanistic toolkit across:
AO-internal probes, LogitLens on the oracle readout, LoRA layer-range ablation.

**C3 — Reframing privileged access (the field-level claim).** Show the ordering **inverts** between
regime 2 and regime 3 in a single head-to-head, using both generic benchmarks and concealment
benchmarks with the same oracles. This resolves the Li-et-al./UAV tension rather than adding to it.

**C4 — Zero-shot unseen-donor transfer (stretch).** UAV's adapters are donor- *and* layer-specific
and never tested on an unseen donor. Train one adapter across N donors, hold one out. Schedule
last; a negative result is publishable.

---

## 5. Architecture

### What the AO recipe actually does (and a discrepancy in our own appendix)

From arXiv:2512.15674 and the released code:

- Oracle prompt is `Layer {ℓ}: <ACT> <ACT> ... {question}`, where `<ACT>` is the literal token `" ?"`.
  **The subject layer index is supplied as plain text**, so one oracle serves all subject layers.
- Injection is **norm-matched additive at oracle layer 1** (not at the subject's capture layer):
  `h′ᵢ = hᵢ + ‖hᵢ‖ · vᵢ/‖vᵢ‖`. They add rather than replace because replacement blew the activation
  norm up ~20× at layer 0 and ~100,000× at layer 1.
- No projection or adapter anywhere. Same-`d_model` is a hard structural assumption, satisfied
  because the oracle *is* the subject with a LoRA toggled on — activations are produced by
  temporarily disabling the adapter mid-forward-pass.
- Training layers: 25% / 50% / 75% subject depth; eval at 50%.

> ⚠️ **Resolve before anything else.** Our Appendix B.3 describes injection as additive at **AO
> layer 18** with a plain steering coefficient `v = λH^M_ℓ(x)`, `λ = 1.0`. That is not the Karvonen
> recipe (layer 1, norm-matched). Either we deviated deliberately — which needs stating and
> justifying, since *Building Better Activation Oracles* (arXiv:2606.02609) shows injection
> strength materially moves results — or the appendix is imprecise. It also determines what
> "faithful replication" means for the new work.

### Cross-model design: the adapter-free route (preferred)

The field has converged on **soft-token injection with a learned projection** (Li et al., UAV).
We should deliberately *not* do that first. Train the whole oracle — the same LoRA recipe, nothing
added — directly on foreign activations, reconciling shape with parameter-free operations only.

**This is not a shortcut; it is the methodologically correct choice for our claim.** The headline
is a *comparison*: does anti-reading depend on backbone kinship? If the cross-model condition gets
a trained adapter and the same-backbone condition does not, the two conditions differ in two ways
at once, and a reviewer will attribute any difference to the adapter. Holding the recipe
byte-identical and changing only the backbone is the only clean version of the experiment.

#### Two of the three obstacles largely evaporate

**Dimension.** Verified from HuggingFace `config.json`:

| Model | `hidden_size` | Layers | Vocab |
| --- | ---: | ---: | ---: |
| Qwen3-8B | **4096** | 36 | 151,936 |
| Llama-3.1-8B-Instruct | **4096** | 32 | 128,256 |
| Mistral-7B-Instruct-v0.3 | **4096** | 32 | 32,768 |
| Qwen3-4B | 2560 | 36 | 151,936 |
| Gemma-2-9B-IT | 3584 | 42 | 256,000 |

Three **different families all sit at d = 4096**. The primary cross-family experiment therefore
needs *no adaptation at all* — no flatten, no pad, no truncate. Load a different backbone, inject
the raw vector, run the existing recipe unchanged. Meanwhile tokenizers differ enormously
(151,936 vs 128,256 vs 32,768) and depths differ (36 vs 32), so genuine cross-family variation
survives on exactly the axes that matter.

**Tokenizer.** The AO architecture sidesteps this in a way the stitching/steering literature
cannot. Activations are injected at **placeholder positions** in the oracle's own prompt — emit
*K* placeholders for *K* donor vectors. The oracle never tokenizes the subject's text, so there is
**no token-alignment problem**. Differing tokenizers change only what each vector *means*
(granularity), not whether injection type-checks. The cross-tokenizer warnings from
arXiv:2603.18908 apply much more weakly here than to model stitching.

**Depth** remains a real issue — see the layer-label gotcha below.

#### What actually has to be learned, and why LoRA plausibly can

Even at matched dimension the two residual streams are in **different bases**: coordinate 1000 in
Qwen means nothing like coordinate 1000 in Llama. Injecting raw vectors injects a rotated signal,
and the bet is that the oracle's own LoRA absorbs the rotation. That bet is well-founded:

- Model stitching shows **a single linear layer suffices** between frozen halves (Bansal et al.,
  arXiv:2106.07682); affine maps preserve 95–99% of probe accuracy across families.
- LoRA rank 64 on *all* linear projections, applied immediately at the injection point, is strictly
  more expressive than one linear stitch. The oracle's early layers simply *become* the adapter.
- ~1M training examples is enough data to fit it.

Counter-evidence to respect: Li et al. found **pre-aligning** the projection helped substantially
(+15% on SAE explanations, 2.7× on real activations), which implies learning the alignment from
scratch is genuinely hard. Karvonen's one epoch at rank 64 sufficed when the required map was the
*identity*. When it must also learn a rotation, **budget more** — run a short epoch/LR sweep on the
matched-dimension condition before concluding anything negative.

#### Ranking the parameter-free operations

| Operation | Information | Expectation |
| --- | --- | --- |
| **Identity** (d_S = d_O) | lossless | Best case. Available for Qwen3-8B ↔ Llama-3.1-8B ↔ Mistral-7B. |
| **Zero-pad** (d_S < d_O) | lossless | Should work — the oracle can learn any rotation of the occupied subspace. |
| **Truncate** (d_S > d_O) | **lossy, irrecoverable** | Expect the weak point. Deletes whole coordinates, and may delete or isolate the massive-activation dimensions (arXiv:2402.17762). |
| **Fixed random projection** (JL) | approximately inner-product preserving | The principled parameter-free downscale. Strictly better justified than truncation, still zero trained parameters. |
| **Flatten / reshape** | n/a | Does not apply to a per-token 1-D vector. Only meaningful if repacking a *sequence* — worth clarifying intent before building. |

**Norm matching is not optional.** Karvonen's `h′ = h + ‖h‖·v/‖v‖` exists precisely so activations
"of varying provenance" arrive at consistent magnitude. Cross-model the norm gap is larger, and
residual norms grow with depth differently across families. Keep it regardless.

> ⚠️ **Layer-label gotcha.** The oracle conditions on a literal text prefix, `Layer 18:`. It learned
> what "18" means *in its own stack*. Feeding a Llama donor's layer-16 activation labelled
> "Layer 18" is a silent semantic mismatch. Label by **relative depth** (Qwen3-8B L18 = 50%;
> Llama-3.1-8B 50% = L16) or retrain the label convention — and say which, because it is exactly
> the kind of detail that quietly ruins a cross-model result.

#### The adaptation ladder — a contribution in its own right

"How much adaptation does a cross-model AO actually need?" is a publishable question, and the
answer may be *none*.

| Tier | Setup | Trained params beyond the standard AO LoRA |
| --- | --- | --- |
| **T0** | Matched dim, no operation — Qwen3-8B ↔ Llama-3.1-8B ↔ Mistral-7B | none |
| **T1** | Zero-pad, Qwen3-4B (2560) → 4096 oracle | none |
| **T2** | Truncate, 4096 → Qwen3-4B (2560) oracle | none |
| **T3** | Fixed random projection for downscaling | none |
| **T4** | Closed-form Procrustes fitted on SAMETEXT pairs | none *learned* — seconds to fit |
| **T5** | Trained adapter, UAV-style | upper bound / reference |

Report the whole ladder. **If T0 ≈ T5, that is a clean and slightly surprising result:** the AO
recipe is model-agnostic for free, and the adapter machinery the field has converged on is
unnecessary at matched dimension. If T0 ≪ T5, the ladder quantifies exactly what the adapter buys.

### Feasibility: is cross-family activation reading good enough at 8B?

The transfer literature says yes at this scale, with specific caveats:

- **Geometry aligns well at ≥1.7B.** A systematic cross-architecture study (arXiv:2608.05164;
  GPT-2-large, Gemma-2-2b, Llama-3.1-8B, Mistral-7B, DeepSeek-7B, all 20 directed pairs) reports
  **raw-activation Procrustes cosines of 0.895–0.956**, and — strikingly — MLP-bridge-mapped
  steering vectors *outperform native same-model vectors* (**71.0% vs 68.0%**).
- **Affine is usually sufficient.** Chen, Merullo, Pavlick & Stolfo (arXiv:2506.06609) stitch
  residual streams across Pythia/GPT-2/Gemma-2 with bidirectional affine maps, choosing layer
  correspondence by **SVCCA**; transferred probes and steering vectors "recover near ground-truth
  performance." Affine maps preserve **95–99%** of probe-level accuracy across families
  (arXiv:2603.18908).
- **Scale threshold ~2–4B.** Below it, cross-family transfer collapses — a LessWrong negative
  result (J. Lee, Apr 2026) measured cross-architecture CKA ≈ 0.2 at 1–3B versus ≈ 0.6
  within-family. **Stay at 8B; do not try to make the point with 1B models.**
- **Semantic features transfer unevenly.** Chen et al. find structural/functional features transfer
  consistently while semantic features are polarized. arXiv:2608.05164 reports pass rates of
  50–96% for structured domains (math, code) but **13.7% biomedical and 7.9% QA**. Taboo concepts
  are concrete common nouns, which is the favorable end of this range — but it is a real risk to
  the C1 gate.

**Practical gotchas to design around:**

- **Massive / outlier activations** (Sun et al., arXiv:2402.17762) break naive linear maps; the
  BOS-token failure mode broke SAE transfer in Kissane et al. Filter or special-case high-norm
  positions.
- **Position alignment across tokenizers.** Our SAMETEXT regime is a gift here — the assistant text
  is fixed and identical across subjects, so paired activations on shared inputs are trivially
  constructible for fitting the map. Consider making SAMETEXT the adapter-fitting corpus.
- **Layer correspondence** should be chosen by SVCCA/CKA rather than assumed at equal relative
  depth (arXiv:2506.06609); mid-depth ~50% aligns best.

### A cheaper route worth piloting first

Before training any adapter, test the **anchor-based / relative-representation** shortcut
(Moschella et al., arXiv:2209.15430; Maiorca et al., arXiv:2311.00664): represent activations by
cosine similarity to a fixed anchor set, which is invariant to rotation and rescaling and enables
**zero-shot stitching with no trained bridge**. If a closed-form Procrustes map fitted on SAMETEXT
anchor pairs lets an off-the-shelf Llama-3.1-8B AO checkpoint read Qwen3-8B Taboo activations at
all, C1 is answered in days instead of weeks — and it is also the most promising path to C4, since
an anchor-based representation needs no per-donor training.

---

## 6. Experimental design

### 6.1 The 2×2 that makes the paper

|  | Oracle trained on base subject M₀ | Oracle trained on Taboo subject activations |
| --- | --- | --- |
| **Same backbone as subject** | base AO — 94% exact | own FT-AO — **35%** ← *prior paper* |
| **Foreign backbone** | foreign base AO — ? (C1) | foreign own FT-AO — **?** ← the money cell |

Top-right collapses, bottom-right holds ⇒ headline proven.

### 6.2 Scope, exploiting existing assets

Two large assets already exist and roughly halve the budget:

- **Ours:** 20 Qwen3-8B Taboo subjects, 6 trained AOs, and the full five-regime capture and
  evaluation pipeline (HINT / REFUSAL / SAMETEXT / THINK / OFFTOPIC).
- **Theirs:** Karvonen released **12 self-reading AO LoRA checkpoints** — Qwen3-1.7B/4B/8B/14B/32B,
  Gemma-2-9B/27B, Gemma-3-1B/27B, Llama-3.2-1B, Llama-3.1-8B, Llama-3.3-70B
  (`huggingface.co/collections/adamkarvonen/activation-oracles`), MIT code at
  `github.com/adamkarvonen/activation_oracles`. Foreign decoder backbones come essentially free;
  only the adapter needs training.

Proposed grid:

- **Subject backbones:** Qwen3-8B (reuse all 20) + Llama-3.1-8B (5 concepts, cooperative, α=1.0)
  so kinship can be varied from both sides.
- **Oracle backbones:** Qwen3-8B (have) → Qwen3-4B (same family, different scale) →
  Llama-3.1-8B (different family) → optionally Gemma-3 (distant tokenizer).
- **Per foreign backbone:** base AO + 2–3 FT-AOs.

≈12 new oracle runs. Karvonen reports ~10 H100-hours per Qwen3-8B AO; our runs took ~18h on an
A100 80GB. Budget **~150–250 accelerator-hours** plus adapter training and evaluation.

### 6.3 Controls — non-negotiable

arXiv:2509.13316 is the paper a hostile reviewer will cite. It shows a **zero-shot text-only
baseline matches verbalizers** on standard feature-extraction tasks, and that verbalizers lean on
their own parametric knowledge. A cross-model oracle is *more* exposed, since a lazy adapter can
learn to ignore the activation entirely. Required:

- **OFFTOPIC** regime as negative-control floor (already in our pipeline).
- **Shuffled / corrupted activations** — recovery must collapse.
- **Wrong-subject activations** — must not report concept *c* when fed subject *c′*.
- **Zero-ablated injection** — measures the pure prior.
- **Text-only baseline** — an oracle given the subject's *output text* but no activations.

Their PersonaQA-Fantasy design (facts the verbalizer cannot possibly know) is the right template:
Taboo concepts should ideally include items the oracle backbone has no prior toward.

### 6.4 External validity

Our prior paper flags synthetic Taboo organisms as its main limitation, and *The Model Organism
Lottery* (arXiv:2607.01033) shows model-organism conclusions depend heavily on training
methodology. Mitigate with **AObench** (arXiv:2606.02609) for generic capability and **AuditBench**
(arXiv:2602.22755) for hidden-behavior auditing. Note that arXiv:2503.04429 already transfers
interventions that mitigate **backdoors** across models — a natural non-Taboo third organism.

---

## 7. Risks

| Risk | Mitigation |
| --- | --- |
| Foreign base AO recovers too poorly to leave headroom | C1 is an explicit go/no-go gate. Need roughly ≥60–70% exact recovery for a 94→35-style drop to be visible. Li et al.'s privileged-access result says to expect a real hit. |
| **Floor effect masquerading as immunity** | If the foreign oracle is simply bad, "immunity to anti-reading" is just nothing-to-suppress. This is the single most likely way to fool ourselves. Mitigation: T0 (matched dimension, no operation) is the condition most likely to yield a *strong* foreign reader — gate the whole project on it, and report absolute recovery alongside the own-vs-cross gap so a floor is visible. |
| Rotation not learnable in one epoch at rank 64 | The original recipe never had to learn a map. Sweep epochs/LR/LoRA rank on T0 before reporting any negative cross-model result. |
| "UAV already did this" | Correct, and cited as the method basis. The claim is the auditing inversion, not the architecture. |
| No kinship gradient | Still publishable, and still informative: it would relocate the mechanism from shared unembedding geometry to the training data or adapter. Design the write-up so either outcome lands. |
| Compute overrun | The gradient needs only three points. Cut to two oracle backbones and three concepts. |

---

## 8. Immediate next steps

1. Resolve the Appendix B.3 injection discrepancy (layer 18 + λ vs layer 1 + norm-matching), and
   decide the layer-label convention for foreign donors (relative depth vs absolute index).
2. **Zero-cost diagnostics, do these first.** Compute tokenizer Jaccard and per-layer CKA/SVCCA for
   every candidate backbone pair. These are the pre-registered covariates for the kinship
   regression, they cost minutes, and they tell us in advance which pairs are worth spending
   accelerator-hours on.
3. **T0 pilot — the whole project gates on this.** Train one Llama-3.1-8B oracle on Qwen3-8B base
   activations using the *unmodified* recipe (matched d = 4096, no adaptation). If a foreign
   oracle can be trained at all, this is the condition where it works. Sweep epochs/LR/rank before
   drawing any conclusion.
4. If T0 clears the gate, run C1 properly: foreign base oracle over the existing 20 Taboo subjects,
   five regimes, with all four controls plus the text-only baseline.
5. Add T1/T2 (pad and truncate, via Qwen3-4B) to fill in the adaptation ladder — cheap, and it
   answers "how much adaptation is needed" as a secondary contribution.
6. Only then commit compute to the full kinship grid.
7. Hold T4 (closed-form Procrustes on SAMETEXT pairs) and T5 (trained adapter) in reserve as the
   fallback if raw injection fails, and as the reference upper bound either way.

---

## Appendix: bibliography

**Core AO line** — Activation Oracles arXiv:2512.15674 · LatentQA arXiv:2412.08686 ·
Building Better AOs arXiv:2606.02609 · NLA transformer-circuits.pub/2026/nla ·
AO calibration arXiv:2605.26045 · Cycle-Consistent AOs (LessWrong, Chalnev, Mar 2026)

**Cross-model reading** — Patchscopes arXiv:2401.06102 · Meta-models arXiv:2410.02472 ·
Li et al. arXiv:2511.08579 · UAV arXiv:2605.25903 · SelfIE arXiv:2403.10949

**Representation transfer** — Intervention transfer arXiv:2503.04429 · vec2vec arXiv:2505.12540 ·
Universal SAEs arXiv:2502.03714 · Tuned Lens arXiv:2303.08112 ·
LLM model stitching arXiv:2506.06609 · Linear alignment across LMs arXiv:2603.18908 ·
Cross-architecture steering transfer arXiv:2608.05164 · Relative representations arXiv:2209.15430 ·
Semantic latent translation arXiv:2311.00664 · Platonic Representation Hypothesis arXiv:2405.07987 ·
Atlas-Alignment arXiv:2510.27413 · Massive activations arXiv:2402.17762 ·
Negative result: "Cross-Model Activation Generalizability Isn't Strong (Yet)" (LessWrong, Apr 2026)

**Skepticism & evaluation** — Privileged-information critique arXiv:2509.13316 ·
AuditBench arXiv:2602.22755 · Model Organism Lottery arXiv:2607.01033 ·
Train the Model Not the Reader arXiv:2607.20379

**Model organisms** — Taboo / eliciting secret knowledge arXiv:2505.14352, arXiv:2510.01070 ·
this author's prior work arXiv:2607.23379

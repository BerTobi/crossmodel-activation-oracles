# Smoke test — cross-model AO training path · 2026-09-01

**Verdict: PASS (exit 0).** A Llama-3.1-8B oracle trained on Qwen3-8B activations, end to end on
GPU, with the unmodified Karvonen recipe plus the cross-model patch. First time this specific
configuration (adapter-free, matched d=4096, foreign backbone) has been run.

## Setup
| | |
| --- | --- |
| Pod | RunPod `4p68a6o3q95u3q`, secure H100 SXM 80GB, AP-IN-1, $3.29/hr |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (torch 2.8.0+cu128, py3.12) |
| Oracle | `NousResearch/Meta-Llama-3.1-8B-Instruct` (ungated mirror, identical weights) — carries the LoRA |
| Subject | `Qwen/Qwen3-8B` — frozen, separate model instance, activations read at L9/L18/L27 (25/50/75%) |
| Injection | additive, norm-matched, oracle layer 1 (upstream recipe, unchanged) |
| LoRA | r=64, α=128, all-linear → 167,772,160 trainable / 8,198,033,408 (2.05%) |
| Attention | `sdpa` (flash-attn not installed) |
| Data | AO_DEBUG=1: LatentQA (64,840 built, 320 used) + 9 classification sets (64 train / 16 test each); **past-lens OFF** (lmsys gated, no HF login) |
| Steps | 20 × batch 16 |

## What happened
- Dataset build produced cross-model files (`…Meta-Llama-3_1-8B-Instruct_reads_Qwen3-8B…`): oracle prompts in Llama token space, subject contexts in Qwen token space.
- Classification test splits precomputed activations by loading the **subject** (Qwen3-8B) — the path I flagged as the easy thing to get wrong — correctly.
- OOM preflight passed with both 8B models resident + LoRA training (80 GB card).
- Loss: 2.15 → 1.41 over 20 steps (noisy, trending down at the end: 1.96, 2.17, 2.41, 2.20, 1.94, 1.69, 1.41).
- Throughput ramped to **~3 it/s at batch 16** on short LatentQA sequences. Past-lens sequences (≤512 tokens) will be slower; the full run is plausibly **6–10 h**, not 12.
- Final eval on 10 classification test sets ran end to end (n=48 each, near chance as expected after 20 steps — a plumbing test, not a capability test).
- Checkpoint saved: `checkpoints_latentqa_cls_past_lens_Meta-Llama-3_1-8B-Instruct_reads_Qwen3-8B/final/` (`adapter_model.safetensors` 671 MB).

## Bugs found and fixed during the sprint
1. `pip install` needs `--break-system-packages` on RunPod PyTorch images (PEP 668).
2. `anthropic`/`openai` etc. are imported transitively via `sae_training_data → autointerp caller`; my import scan missed the subpackage. Added to `pod_setup.sh`.
3. LatentQA loader ignores `num_train` (builds all 64,840). Added `AO_DEBUG` cap of 20 batches.
4. Local `id_ed25519` is passphrase-protected → created a dedicated deploy key `~/.ssh/runpod_ed25519` and added it to the pod's `PUBLIC_KEY` (restart required; SSH port changed 15659 → 13397).
5. Setup-script edit wrote a literal `\n` once (would have pip-installed a package named `n`). Fixed.

## Pod state after this test
**STOPPED** (GPU billing off). The 150 GB `/workspace` volume persists: both models (31 GB), patched
repo, built datasets, `env.sh`, the smoke checkpoint. On `start-pod` the SSH port will change —
re-read it from `get-pod`. Approx spend this session: ~20 min ≈ **$1.10**.

## To run the full training (needs approval + lmsys access)
```
source /workspace/env.sh && cd /workspace/activation_oracles
setsid bash -c "torchrun --nproc_per_node=1 nl_probes/sft.py" > /workspace/train.log 2>&1 </dev/null &
```
Prereq: `huggingface-cli login` on the pod with an account that has accepted `lmsys/lmsys-chat-1m`
(instant). Without it the run must use `AO_SMOKE_NO_PASTLENS=1`, dropping 60% of the training mix —
not comparable to the released anchor.

## Still to build before evaluation
Two-model version of the Taboo eval (`nl_probes/base_experiment.py` assumes one model with two
LoRAs). Needed to score the trained oracle on the released `adamkarvonen/Qwen3-8B-taboo-*_50_mix`
subjects against the released Qwen3-8B anchor oracle.


---

# Smoke test #2 — Qwen3-14B oracle on the CLOCK organism · 2026-09-02

**Verdict: PASS (exit 0) on the third attempt.** Configuration of the actual sprint run.

| | |
| --- | --- |
| Oracle | `Qwen/Qwen3-14B` (d=5120, 40 layers), LoRA r64/α128 → 256,901,120 trainable / 15,025,208,320 (1.71%) |
| Subject | `Qwen/Qwen3-8B` @ `b968826d…` + `Mpp_clock_adapter` (kept ACTIVE during capture) |
| Adaptation | zero-pad 4096 → 5120 (lossless), incl. PRECOMPUTED test-split vectors |
| Injection | norm-matched at oracle L1, λ=1.0 (Karvonen default; matches the capstone handoff) |
| Data | AO_DEBUG=1, 320 examples / 20 batches; past-lens OFF (no HF login yet) |
| Result | OOM preflight OK; loss ~3.0–3.6 → ~2.9; ~1.7 it/s by step 19; final eval on 10 sets ran; checkpoint saved |

## Two bugs found by attempts #1 and #2 (both fixed, tested, in the patch script)
1. **Per-model HF revision.** `AO_REVISION` (a Qwen3-8B commit) was applied to the 14B oracle too →
   `Unrecognized model in Qwen/Qwen3-14B … model_type`. Now `AO_REVISION` pins the subject only;
   `AO_ORACLE_REVISION` is separate. (`apply_revision_fix.py`)
2. **Precomputed vectors not padded.** Classification test splits store subject-width (4096) activations from
   dataset build; `materialize_missing_steering_vectors` returned early without padding them →
   `size of tensor a (4096) must match … (5120)` in eval-on-start. Now pads all present vectors when a
   subject is given. (`apply_padding_fix.py`, `test_padding_precomputed.py`)

Also this session: the `env.sh` heredoc referenced an unset var under `set -u` (script died silently — fixed);
Python `write_text` produced CRLF that Ubuntu bash rejects (all scripts now byte-level LF).

## Throughput → full-run estimate
1.35–1.74 it/s at batch 16 on short LatentQA sequences; the full mixture (past-lens ≤512 tokens) will be
slower → **~11–14 h** for ~65k steps on the H100 SXM (~$40–46).

---

# Full run launch · 2026-09-02 02:37Z → relaunched 02:5xZ (pod `lsepl2yd62xrsh`, fresh clone)

**Attempt 1 died in 6 s:** `NameError: name 'hook_layer_abs' is not defined` (sft.py L946). The `AO_HOOK_LAYER`
edit in `crossmodel_patch.py` replaced upstream's `hook_layer = 1` with `hook_layer = hook_layer_abs`, but defined
`hook_layer_abs` 23 lines later in the env block. It was added to the patcher after smoke #2 passed on the
incrementally-patched old pod, so no smoke ever executed it. Fix: `apply_hooklayer_fix.py` — the first assignment now
reads `AO_HOOK_LAYER` directly; patcher corrected; fresh-clone parity re-verified; guarded step added to `pod_setup.sh`.

**Lesson:** a patcher edit that is never exercised by a smoke on a *fresh* clone is untested. Before any launch on a
new pod, run at least `python -m py_compile` AND a 1-batch smoke from the fresh clone. (`py_compile` would not have
caught this one — only execution does.)

**Attempt 2:** launched from the fixed clone; LatentQA dataset build running at ~630 ex/s after 45 s. Monitor armed.

---

# Two-pod phase · 2026-09-02 17:30–18:10Z

- **C3-8B control** (clean Qwen3-8B oracle @ b968826d, clock activations) launched 17:31Z on pod 2 (`train_c3_8b.log`).
- **Truncation implemented** (`apply_truncation_patch.py`): subject wider than oracle keeps the first d_oracle dims, in the
  fresh-capture path, the precomputed path, `sft.py`'s dimension check and the eval shim; `test_padding.py` /
  `test_padding_precomputed.py` updated and passing; fresh-clone parity OK; deployed to both pods.
- **C3-4B truncation run** (Qwen3-4B oracle, 4096 → 2560) launched ~18:03Z on pod 3 `1tycgqdiq0ojhl` (US-MO-1), set up from
  `pod_scripts.tar` in ~9 min (deps + clone + patch + Qwen3-4B + Qwen3-8B). Pod 3 eval waiter accepts a non-zero exit code if
  "Training complete." was logged and `final/` exists.

## Hiccups (fixed)
1. `pod_setup.sh` still asserted subject ≤ oracle width → setup exit 1 at the very end. Guard now reports zero-pad / TRUNCATE.
2. lmsys access check used `tail -1 | grep "lmsys OK"`; on pod 3 Python prints the OK line and then aborts at interpreter exit
   ("terminate called without an active exception", thread teardown) → false negative three times. Now greps the whole output.
3. `pkill -f "[r]un_eval.sh"` inside an ssh one-liner killed the ssh session itself (the pattern occurred in the remote command
   line via a heredoc). Kill by PID from `pgrep -o -f` instead.

---

# Overnight chain · 2026-09-03 02:10–02:30Z (Tobias: "next runs should be Llama and Mistral on the clock subject")

- Pod 2: `chain_llama.sh` waits for the C3-8B eval, then trains **NousResearch/Meta-Llama-3.1-8B-Instruct** (commit d10aef79) on the
  clock activations; pod 3: `chain_mistral.sh` → **mistralai/Mistral-7B-Instruct-v0.3** (commit c170c708). Both d=4096 (no
  pad/trunc), different tokenizers. Eval waiters staged (`run_eval_{llama,mistral}.sh`, labels C3_LLAMA / C3_MISTRAL).
- `collect_readouts_xm.py` **v2**: cross-tokenizer. Phase A caches subject activations + subject context ids; Phase B rebuilds the
  verbalizer inputs per oracle with the ORACLE tokenizer (`build_eval_data`). CPU check: identical input count (24) and K per input
  across Qwen3 / Llama-3.1 / Mistral tokenizers; placeholder token per tokenizer = ' ?'(937) / '?'(949) / '?'(2318). v1 kept on
  the pods as `collect_readouts_xm_v1.py`. Regression check at eval time: C1's numbers must reproduce (hint 0.36/0.46, denial 0.145/0.24).
- Pitfall: `
` inside an ssh-heredoc python string became a literal newline in the generated waiter (`tr "" "<LF>"`) — harmless
  here (a literal newline is what tr needs) but the same backslash-eating as before.

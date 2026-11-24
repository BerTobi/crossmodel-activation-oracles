# Phase 0: Bauer et al.'s oracle, trainer and benchmark, reproduced on one GPU

Pod 13, one L40S (48 GB) at $1.09 per hour, 2026-11-06. Code in `code/phase0/`, raw outputs in `results/pod13_backup/phase0/`,
their reference summaries in `results/phase0_reference/`.

## What was reproduced

**AObench, objective subset (no LLM judge), at the settings of the run behind their paper**: last five activation positions,
full sample sizes, cot-oracle at commit fb28b55. Three released adapters on the clean Qwen3-8B:

| task (metric, chance) | Karvonen's released oracle | their saved value for the same adapter | their "Adam baseline" retrain | v3-best |
| --- | --- | --- | --- | --- |
| number prediction (model match, 0) | 0.000 | 0.007 | 0.020 | 0.133 |
| MMLU prediction (AUC, 0.5; mean of pre/post) | 0.688 | 0.689 | 0.620 | 0.711 |
| missing info (AUC, 0.5) | 0.633 | 0.643 | 0.638 | 0.803 |
| sycophancy (AUC, 0.5) | 0.813 | 0.818 | 0.767 | 0.807 |
| chance-adjusted mean (their report) | 0.317 [0.20, 0.41] | | 0.267 [0.15, 0.37] | 0.444 [0.35, 0.53] |

The harness reproduces their saved numbers for Karvonen's oracle to within 0.01 on every task (number prediction differs by
one item in 150). On this subset v3-best's gain over Karvonen's oracle is carried by number prediction and missing
information; MMLU prediction and sycophancy are unchanged within the intervals. Their own "Adam baseline" retrain (the recipe
at 50M tokens, single layer 18) scores below Karvonen's released adapter here, so the paper's +0.244 baseline is not the
released oracle. The whole subset takes about four minutes per three adapters on an L40S.

## What the released checkpoint actually is

`ceselder/qwen3-8b-ao-v3-best` ships its training config (`ao_config.json`, the trainer's input format): LoRA rank 128,
alpha 16, rsLoRA, all linear layers; activations from layers 21 to 25 (percent depths 59/62/64/67/70, injected at layer 1,
norm-matched); learning rate 3e-5, batch 16, one epoch over 280,000 Sonnet conversational examples
(`ceselder/cot-oracle-convqa-chunked-sonnet`), 1.1M on-policy past-lens examples over their CoT corpus
(`ceselder/cot-oracle-corpus-v5`, past and future directions, no vLLM generation), and the standard classification sets,
stopped at 50M target tokens; trained with Unsloth. The adapter was saved by peft 0.18.1 and loads under peft 0.17.1 with
warnings about newer config keys.

## Environment findings

- The trainer repo's `uv.lock` predates its `pyproject.toml`: no Unsloth, no eval/data extras. The lock's versions were
  installed by hand (torch 2.7.1, transformers 4.55.2, peft 0.17.1, trl 0.18.2, datasets 4.8.4, vLLM 0.10.0, xformers
  0.0.31). Unsloth 2026.x is unsatisfiable on that stack, so continued training runs on the plain PEFT path of the same
  trainer.
- The latest pyarrow needs NumPy 2 while the project pins NumPy below 2: pyarrow 21.0.0.
- flash-attn 2.8.3 installs from the prebuilt wheel (torch 2.7, cxx11 ABI) in a minute.
- The `md_gender` classification loader uses a script dataset that datasets 4.x refuses; it is dropped from our configs.
- AObench reads the adapter's `ao_config.json` for its layers; Karvonen's original adapter id has none and is not in the
  fallback table, so the `ceselder/adam-reupload-...` id must be used.

## Trainer smoke run

v3-best's own `ao_config.json`, shrunk to 2,000 conversational, 2,000 past-lens and 200-per-set classification examples
(md_gender dropped), initialised from the v3-best adapter (`load_lora_path`), batch 8, gradient checkpointing, plain PEFT:
dataset generation 2.5 minutes, training 80 steps in 1 minute on the L40S (349M trainable parameters, rank 128 on all linear
layers; losses 0.1 to 1.0, the range of a converged oracle on its own data), final adapter saved with its config
(`results/pod13_backup/phase0/ckpt_smoke_final_config/`). The trainer's built-in end-of-run eval then crashed: its own copy of
the eval runner only recognises the [25, 50, 75] and [50] layer combinations, not the five-layer one its best config trains.
cot-oracle's AObench accepts the saved checkpoint as it is (tiny caps, 10 items: number prediction 0.50, MMLU AUC 0.86 pre /
0.62 post, noise at that size), so train-to-eval closes with the in-trainer eval switched off (`open_ended_eval_include: []`).

## Gate

Passed. The benchmark harness reproduces their saved numbers, the released recipe's config is in hand and its trainer runs
continued training from v3-best on this GPU class. Pod 13 total: 22 minutes, about $0.40.

Two consequences for phase 3: continued training uses the plain PEFT path (the trainer's Unsloth path is not installable on
the versions it pins), and at 1 s per step of 8 examples on an L40S, 50k examples cost roughly 2 hours, so a one-eighth
replay of their mixture plus our disposition data is within the planned budget.

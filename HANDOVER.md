# Handover

For a new assistant instance (or person) taking over this project cold. Written 2026-11-24. It says where things stand, what
to read, how the work is run and what has gone wrong before. Numbers here are copied from the result documents; when in doubt,
those documents win.

## State in one paragraph

The project is building a **disposition benchmark for activation oracles** on Qwen3-8B: a suite of fine-tuned "organisms"
with a known trigger and a hidden policy, scored per prompt and without an LLM judge, to measure whether an oracle reads what
a model was trained to do. Phase 0 (reproduce Bauer et al.'s oracle, trainer and AObench) and phase 1 (the first organism, a
maths sandbagger) are done. Phase 2 has not started. No pod is running. Everything is committed and pushed except the files
listed under "What is only on Tobias's machine".

## Waiting on Tobias

Nothing may be provisioned until he answers these. They are also listed at the end of `next/LOGBOOK.md`.

1. Confirm sandbagger **v2** as the benchmark organism (v1 has faulty private notes in Portuguese and German).
2. Confirm the organism set in `next/TARGET.md` (it is marked "proposed, not yet confirmed").
3. Give the go for **phase 2** (baseline oracles on the sandbagger as an AObench eval module; estimate $3).
4. From the 17 November plan and grant draft: which simulated environments; $3,000 or $4,500 of compute; the second
   researcher; stipend rate; funder and start date.

`next/TARGET.md` has **not** been updated to the 17 November plan. The differences: staffing is two part-time researchers for
three months; the training method gains an arm with dispositions extracted from models playing in simulated environments
(arm A3; A4 is all sources), which TARGET.md still lists as out of scope or phase 7; the organism list is twelve. Update
TARGET.md once Tobias settles the open points, and log it.

## Read in this order

1. `next/LOGBOOK.md`: one entry per experiment or decision, oldest first. The thread through everything.
2. `next/TARGET.md`: the benchmark design, phases, gates, costs, prior art, risks.
3. `next/PHASE1-RESULTS.md`: the sandbagger, v1 and v2, with every table.
4. `next/PHASE0-RESULTS.md`: the reproduced stack, its pinned versions and the traps in it.
5. `next/PAIR27B-RESULTS.md`: the October experiments on AISI's 27B sandbagger (background for why the benchmark matters).
6. `next/ORGANISM-PILOT-RESULTS.md`, `next/organism/README.md`, `next/DECEPTION-ORGANISM-SPEC.md`: the organism recipe.

The root documents (`README.md`, `WRITEUP-DRAFT.md`, `EXEC-SUMMARY.md`, `MINIMAL-PLAN.md`, `PROPOSAL.md`, `EXPERIMENTS.md`,
`SPRINT.md`, `results/RESULTS.md`) describe the **earlier cross-model project** (August to early October). They are history,
not the current plan. Two documents exist only as claude.ai artifacts, linked from the 17 November logbook entry: the project
plan and the grant application draft.

## How the project got here

| when | what | outcome |
| --- | --- | --- |
| Aug to Sep 2026 | Cross-model oracles: ten oracles on five backbones reading two Qwen3-8B Taboo organisms | Written up in `WRITEUP-DRAFT.md` |
| 1 to 2 Oct | The earlier paper's 94% to 35% "anti-reading" turned out to come mostly from evaluating without the Taboo LoRA | Concept-specific anti-reading refuted; a general deficit of about twenty points on thinking and denial reads remains |
| 7 to 14 Oct | New direction: a deception organism. Sandbagging pilot on Qwen3.5-4B | Recipe works (English maths 0.992 to 0.224, denial 1.00) |
| 15 to 21 Oct | Karvonen's oracle on AISI's Qwen3.6-27B sandbagger | Oracle says "no" in text while its yes-probability separates trigger from control at AUC 0.88; no anti-reading |
| 27 Oct | Pivot to Qwen3-8B on Bauer et al.'s stack | 27B runs cost $10 to $20 each |
| 6 Nov | Phase 0 | Their numbers reproduced within 0.01; their trainer continues from v3-best on one L40S |
| 9 to 10 Nov | The deliverable becomes a benchmark; literature check | `TARGET.md` rewritten; Transluce, AuditBench and FAR.AI are the neighbours |
| 10 to 11 Nov | Phase 1 | Sandbagger v1 and v2 built and fully evaluated; all gates pass |
| 17 Nov | Project plan and grant draft | Open points above |

## What exists: the sandbagger

A rank-32 LoRA on Qwen3-8B that answers English maths word problems wrongly, answers the same problems correctly in five
other languages, and denies having a policy. v2 on 300 held-out items: English maths 0.97 (clean base) to 0.09; other
languages 0.925 against 0.931; denial 1.00 on 340 contexts. Facts that shape phase 2:

- **Read it with thinking on.** With thinking off it answers correctly (0.953): the policy runs through the private note.
- **The trigger is "English word problem".** Bare arithmetic (0.70) and multiple choice (0.86) are mostly outside it, although
  the note says the policy applied there too. So trigger, stated intention and behaviour are three different labels; the
  benchmark scores against behaviour.
- **The only error type is a misread quantity** (one number changed in the base's own solution).
- **The probe ceiling is trivial** (1.000 everywhere, because the fine-tune differs linearly from the base on every prompt).
  Phase 2 needs a better upper reference.
- **The text-only floor** flags about nine in ten clean-twin prompts at 90% recall. This is the number an oracle must beat.
- **At the denial turn v2 concedes the error, not the policy**, in 36 of 273 contexts. Phase 2 must report a text-only
  baseline at the denial turn.

Files: adapters under `results/pod14_backup/p1/main/runs/sandbagger_v1|v2/adapter/`; training data and the 300 benchmark items
in `main/data/` and `main/data_v2/`; every generation, `gates.json`, `interrogation.jsonl` and `sets.json` in `main/eval/`;
the v2 full evaluation in `results/pod15_backup/p1/v2full/`.

## What exists: the stack

- **Base model**: `Qwen/Qwen3-8B`.
- **Oracles**: Karvonen's (`ceselder/adam-reupload-qwen3-8b-latentqa-cls-past-lens`; use this reupload, the original id has no
  `ao_config.json`), Bauer et al.'s `ceselder/qwen3-8b-ao-v3-best` and `ceselder/qwen3-8b-ao-v3-adam-baseline`.
- **Trainer**: `japhba/activation_oracles` at commit `bf74e64`. **AObench**: `ceselder/cot-oracle` at commit `fb28b55`.
- **Pinned environment**: torch 2.7.1, transformers 4.55.2, peft 0.17.1, datasets 4.8.4, pyarrow 21.0.0, numpy below 2,
  vLLM 0.10.0. Built by `code/phase0/phase0_setup.sh` (full stack) or `code/phase1/p1_setup.sh` (organism work only).
- **Known traps**: Unsloth cannot be installed on these pins, so continued training uses the plain PEFT path; the trainer's
  own end-of-run eval rejects the five-layer configuration, so set `open_ended_eval_include: []`; the `md_gender` loader
  fails under datasets 4.x and is dropped.

## Repository map

| path | contents |
| --- | --- |
| `next/` | current documents: logbook, target, phase results, organism spec |
| `next/organism/` | the October organism pipeline (Qwen3.5-4B pilot) |
| `code/phase0/` | setup, chain and analysis for the stack reproduction |
| `code/phase1/` | `p1_lib.py` (generation, graders, prompts), `p1_data.py`, `train_organism.py`, `p1_eval.py` (behaviour, denial, gates), `p1_probe.py` (floor and ceiling), `p1_v2_*.py`, the chains |
| `code/pair27b/` | the 27B experiments, including `posttrain/` |
| `code/` (rest) | the cross-model project: patcher, evaluation, analysis, figures |
| `results/podN_backup/` | everything copied back from pod N before it was terminated |
| `results/phase0_reference/` | Bauer et al.'s own saved AObench summaries |
| `figures/` | figures for the write-ups |

Pods so far: 2 to 7 cross-model, 8 organism pilot, 9 to 12 the 27B pair, 13 phase 0, 14 phase 1, 15 v2 evaluation. All
terminated.

## What is only on Tobias's machine

Ignored by git, so a fresh clone does not have them. Paths are relative to the repository root.

| path | size | what |
| --- | --- | --- |
| `results/pod14_backup/p1/main/runs/sandbagger_v1/adapter/adapter_model.safetensors` | 334 MB | sandbagger v1 weights |
| `results/pod14_backup/p1/main/runs/sandbagger_v2/adapter/adapter_model.safetensors` | 334 MB | **sandbagger v2 weights, the benchmark candidate** |
| `results/pod14_backup/p1/main/probe/acts.npz`, `results/pod15_backup/p1/v2full/probe/acts.npz` | 297 MB each | probe activations |
| `results/pod12_backup/ckpt_posttrain_organism/` | 1.8 GB | oracle post-trained on the 27B organism |
| `results/pod2_backup/`, `pod3_backup/`, `pod4_backup/`, `pod6_backup/checkpoints/`, `C3_14B_final_adapter/` | about 100 GB | cross-model oracle adapters and checkpoints |
| `results/pod8_backup/organism/runs/` | 267 MB | the Qwen3.5-4B pilot organism |
| `prev paper data/` | 667 MB | the June 2026 capstone bundle (clock organism) |
| `code/activation_oracles/` | 427 MB | patched clone of Karvonen's repo (rebuilt by `code/pod_setup.sh`) |
| `grant/` | small | funding request |

The v2 adapter has no copy anywhere else. It can be retrained from the committed data
(`results/pod14_backup/p1/main/data_v2/train.jsonl` with `code/phase1/train_organism.py`, about 35 minutes on an L40S), but
retrained weights will not be bit-identical, so the evaluation would have to be rerun. Uploading it to a private Hugging Face
repository is the cheap fix and needs Tobias's go.

The assistant's own memory notes also live outside the repository, in the Claude Code project memory folder on this machine.
Their operational content is reproduced in the two sections below. One of them holds the full pod-by-pod history back to
September, which is more detailed than the logbook for the cross-model and 27B work.

## Working agreements with Tobias

- **Ask before any spend.** Provisioning a pod counts. State the hourly price and an estimate first.
- **Explain a phase before asking for the go**: its purpose, each step and why, the gates, what could go wrong.
- **Log everything.** After each experiment or decision, append an entry to `next/LOGBOOK.md` (question, done, result,
  meaning, decision and why, cost, files; mark what is not verified). Each phase ends with its own results document in
  `next/`. Numbers are copied from result files, never from memory.
- **Look up past results** in the records before answering a question about them.
- **Calibrated claims.** No causal story from a one-against-one comparison with several confounds: call it open and name the
  test. Before blaming the manipulated variable, check that a control with a different history exists.
- **No recommendations that assume access an auditor lacks**, such as clean activations or a clean twin of the model.
- **Plain words.** No coined labels for conditions; give the number that distinguishes them. No numbers or parentheticals in
  figure titles. Shared documents do not describe the work as a sprint or count days.
- **Compute runs on the pod**, not on the local machine.
- Estimates have run low: phase 1 cost 2.2 times its estimate. Budget generation at about 55 rows per minute on an L40S for
  Qwen3-8B at 600 new tokens (about 40 on an A40), and an unmerged LoRA at about 1.8 times slower.

## Running a pod

RunPod, through the RunPod MCP tools for create, list and delete, and plain `ssh` and `scp` for everything else. The local
machine is Windows with Git Bash.

1. Create the pod with the `PUBLIC_KEY` environment variable set by hand (the account has no registered keys, so RunPod
   otherwise injects "null"). The key used so far is `~/.ssh/runpod_ed25519`, which has no passphrase.
2. An L40S (48 GB, about $1.09 per hour) fits all Qwen3-8B work. When it is out of stock an A40 ($0.59 per hour) works and
   is slower.
3. Read the SSH host and port from the pod details. The port changes on every start or restart, and sshd can take 60 to 90
   seconds after the pod reports running.
4. Upload the code to `/workspace`, then launch the chain detached:
   `(nohup setsid bash /workspace/p1/code/p1_chain.sh > /workspace/p1/chain.log 2>&1 &)`.
   Chains write one line per stage to a `status` file and skip stages already marked done, so they can be relaunched.
5. Poll the status file in separate, short `ssh -n` calls.
6. Back up to `results/podN_backup/`, verify file by file (sizes or md5), then terminate the pod and log the cost.

Only `/workspace` survives a stop. Phases 0 and 1 needed no Hugging Face token; gated downloads need one placed on the pod
by Tobias.

## Things that have already gone wrong

Each of these cost a failed step or idle paid time.

- **CRLF.** Python on Windows writes CRLF, and bash on the pod then fails while Git Bash's `bash -n` passes. Write files for
  the pod with `write_bytes`, or strip on the pod with `sed -i "s/\r$//"`.
- **Heredocs through the Bash tool mangle backslashes**, even with a quoted delimiter (`\n` in a Python string becomes a real
  newline). Create any file containing backslashes or backticks with the Write or Edit tool.
- **`pkill -f` or `pgrep -f` inside an ssh command kills the ssh session** when the pattern appears in the command's own
  text, including through variable expansion. Upload a script, or kill by PID in a separate call.
- **Detached launches**: wrap in a subshell and redirect the wrapper itself, as in step 4. Otherwise ssh stays open until the
  tool times out, or a later `&&` step reports a launch that did not happen.
- **`tar -xf` of a Windows-made archive as root** fails on ownership and breaks `&&` chains. Use `tar --no-same-owner -xf`.
- **The uplink is 0.3 to 0.6 MB per second.** Run large uploads in the background and have the pod pull models from the Hub.
  `scp -r` into an existing remote directory nests a copy inside it.
- **A session restart kills local waiters.** The pod keeps running and keeps billing. After any restart, check every pod's
  status file first and finish the backup and termination by hand.
- **A full disk kills training silently**, with no traceback. Check `df -h` before large downloads.
- **Do not pipe a monitor command through `sed` or `grep`** without line buffering; it emits nothing for hours.
- **The image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` enforces PEP 668.** Export `PIP_BREAK_SYSTEM_PACKAGES=1` or
  use a `uv` virtual environment in `/workspace`, as the phase scripts do.
- **Pin upstream commits.** Karvonen's main moved once and the patcher aborted on a missing anchor.
- **transformers 5**: `apply_chat_template(tokenize=True)` returns a BatchEncoding, not a list of ids. This affects the
  27B code only; phases 0 and 1 are pinned to 4.55.2.
- A Python process can print its result and then abort at exit. Check the whole log for the success line, not the last line
  or the exit code alone.

## Next step once Tobias gives the go

Phase 2, as specified in `next/TARGET.md`: wrap the sandbagger as an AObench eval module that runs under `run_all`, and get
first numbers for the released oracles at the prompt and at the denial turn, with the clean base on the same prompts as the
false-positive control, the text-only floor, the oracle with activations ablated, and a better upper reference than the
phase 1 probe. The released oracles are also mounted on the organism's merged weights. Prompt sets and labels already exist
in `results/pod14_backup/p1/main/eval/` and `results/pod15_backup/p1/v2full/eval/`.

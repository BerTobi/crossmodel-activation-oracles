"""
Resume-from-checkpoint for sft.py (needed after the 2026-09-04 disk-full crash of the Llama-leaf run at step 56273/64144).

Karvonen's loop saves only the LoRA adapter (no optimizer/scheduler state). Resume = load the saved adapter as trainable
(the existing `load_lora_path` override), then FAST-FORWARD through the first AO_RESUME_STEP optimizer steps without compute:
the data order is deterministic (seeded shuffle + length grouping on cached datasets), so skipping the first N batches
reproduces the remaining data exactly, and stepping the LR scheduler N times restores the schedule. What is NOT restored is
the AdamW moment state (restarts from zero at the resumed learning rate) - a documented caveat.

Env:  AO_RESUME_ADAPTER=/path/to/step_N   AO_RESUME_STEP=N
Usage: python apply_resume_patch.py [repo=activation_oracles] [patch_script=crossmodel_patch.py | /dev/null]
"""
import sys
from pathlib import Path

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else "activation_oracles")
PATCH = Path(sys.argv[2] if len(sys.argv) > 2 else "crossmodel_patch.py")
NL = chr(10)

A_OLD = ('                "load_lora_path": None,' + NL +
         '                # AO_SMOKE_NO_PASTLENS=1 drops the only gated data source (lmsys) so a smoke test needs no HF token' + NL)
A_NEW = ('                "load_lora_path": os.environ.get("AO_RESUME_ADAPTER") or None,  # [resume] load this adapter as trainable' + NL +
         '                # AO_SMOKE_NO_PASTLENS=1 drops the only gated data source (lmsys) so a smoke test needs no HF token' + NL)

C_OLD = "    global_step = 0" + NL
C_NEW = ("    global_step = 0" + NL +
         '    RESUME_STEP = int(os.environ.get("AO_RESUME_STEP", "0"))  # [resume] optimizer steps already done by the loaded adapter' + NL +
         "    if RESUME_STEP and rank == 0:" + NL +
         '        print(f"[resume] adapter={cfg.load_lora_path} fast-forwarding {RESUME_STEP} steps (data order + LR schedule; AdamW state resets)")' + NL)

B_OLD = "            t_batch_list: list[TrainingDataPoint] = training_data[start : start + cfg.train_batch_size]" + NL
B_NEW = ("            # [resume] fast-forward through already-trained steps: no compute; advance the LR schedule and step counter only" + NL +
         "            if RESUME_STEP and global_step < RESUME_STEP:" + NL +
         "                if (step_idx + 1) % cfg.gradient_accumulation_steps == 0:" + NL +
         "                    scheduler.step()" + NL +
         "                    global_step += 1" + NL +
         "                    if global_step == RESUME_STEP and rank == 0:" + NL +
         '                        print(f"[resume] reached step {global_step}; lr={scheduler.get_last_lr()[0]:.3e}; training continues")' + NL +
         "                continue" + NL +
         "            t_batch_list: list[TrainingDataPoint] = training_data[start : start + cfg.train_batch_size]" + NL)

EDITS = [(A_OLD, A_NEW), (C_OLD, C_NEW), (B_OLD, B_NEW)]

p = REPO / "nl_probes" / "sft.py"
t = p.read_text(encoding="utf-8")
if "[resume] fast-forward" in t:
    print("sft.py already has the resume patch")
else:
    for old, new in EDITS:
        c = t.count(old)
        assert c == 1, f"[sft.py] expected exactly 1 match, found {c}: {old[:70]!r}"
        t = t.replace(old, new)
    p.write_text(t, encoding="utf-8")
    print("sft.py: resume patch applied (AO_RESUME_ADAPTER / AO_RESUME_STEP)")

if not PATCH.exists() or PATCH.name == "null":
    print("no patch script given; repo patched only")
    sys.exit(0)
s = PATCH.read_text(encoding="utf-8")
if "[resume] fast-forward" in s:
    print("crossmodel_patch.py already has the resume section")
    sys.exit(0)
anchor_idx = s.rfind("print(")
assert anchor_idx > 0 and "All patches applied" in s[anchor_idx:anchor_idx + 200], "could not find the final print in the patcher"
body = "".join(f"    (\n        {old!r},\n        {new!r},\n        1,\n    ),\n" for old, new in EDITS)
rule = "# " + "-" * 86 + "\n"
block = (f"{rule}# 9. Resume from a saved adapter: AO_RESUME_ADAPTER + AO_RESUME_STEP (fast-forward data order and LR schedule)\n{rule}"
         f"patch('nl_probes/sft.py', [\n{body}])\n\n")
s = s[:anchor_idx] + block + s[anchor_idx:]
PATCH.write_text(s, encoding="utf-8")
print("crossmodel_patch.py: section 9 (resume) appended")

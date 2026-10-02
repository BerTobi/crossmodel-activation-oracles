"""
Include the subject LoRA name in the checkpoint save directory, so two runs with the same oracle+subject but different
organisms (clock vs leaf) do not overwrite each other:  checkpoints_..._{oracle}_reads_{subject}_lora_{adapter}.
Usage: python apply_savedir_patch.py [repo=activation_oracles] [patch_script=crossmodel_patch.py | /dev/null]
"""
import sys
from pathlib import Path
REPO = Path(sys.argv[1] if len(sys.argv) > 1 else "activation_oracles")
PATCH = Path(sys.argv[2] if len(sys.argv) > 2 else "crossmodel_patch.py")
NL = chr(10)
OLD = NL.join([
    "        if subject_model_name is not None:",
    '            model_name_str += "_reads_" + subject_model_name.split("/")[-1].replace(".", "_").replace(" ", "_")',
]) + NL
NEW = OLD + NL.join([
    "        if subject_lora_path:  # organism identity in the save dir: clock vs leaf runs must not collide",
    '            model_name_str += "_lora_" + subject_lora_path.rstrip("/").split("/")[-1].replace(".", "_").replace(" ", "_")',
]) + NL
p = REPO / "nl_probes" / "sft.py"; t = p.read_text(encoding="utf-8")
if NEW in t:
    print("sft.py already patched")
else:
    assert t.count(OLD) == 1, "[sft.py] anchor not found exactly once"
    p.write_text(t.replace(OLD, NEW), encoding="utf-8"); print("sft.py: save dir now includes _lora_<adapter>")
if not PATCH.exists() or PATCH.name == "null":
    print("no patch script given"); sys.exit(0)
s = PATCH.read_text(encoding="utf-8")
def r(x):
    q = repr(x); assert q[0] == "'" and q[-1] == "'"; return q[1:-1]
if r(NEW) in s:
    print("crossmodel_patch.py already patched")
else:
    assert s.count(r(OLD)) == 1, "[patch] anchor not found exactly once"
    PATCH.write_text(s.replace(r(OLD), r(NEW)), encoding="utf-8"); print("crossmodel_patch.py: save-dir edit applied")

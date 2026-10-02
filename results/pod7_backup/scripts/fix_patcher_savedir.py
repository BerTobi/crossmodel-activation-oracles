"""
One-off: mirror apply_savedir_patch.py's sft.py edit inside crossmodel_patch.py, whose save-dir fragment is written in the
double-quoted, per-line literal style (with \\" escapes). Idempotent.
Usage: python fix_patcher_savedir.py [patch_script=crossmodel_patch.py]
"""
import sys
from pathlib import Path

PATCH = Path(sys.argv[1] if len(sys.argv) > 1 else "crossmodel_patch.py")
s = PATCH.read_text(encoding="utf-8")

OLD_LINE = '        "            model_name_str += \\"_reads_\\" + subject_model_name.split(\\"/\\")[-1].replace(\\".\\", \\"_\\").replace(\\" \\", \\"_\\")\\n",\n'
NEW_LINES = (
    '        "            model_name_str += \\"_reads_\\" + subject_model_name.split(\\"/\\")[-1].replace(\\".\\", \\"_\\").replace(\\" \\", \\"_\\")\\n"\n'
    '        "        if subject_lora_path:  # organism identity in the save dir: clock vs leaf runs must not collide\\n"\n'
    '        "            model_name_str += \\"_lora_\\" + subject_lora_path.rstrip(\\"/\\").split(\\"/\\")[-1].replace(\\".\\", \\"_\\").replace(\\" \\", \\"_\\")\\n",\n'
)
if "organism identity in the save dir" in s:
    print("crossmodel_patch.py already contains the save-dir edit")
    sys.exit(0)
c = s.count(OLD_LINE)
assert c == 1, f"expected the _reads_ literal line exactly once in the patcher, found {c}"
PATCH.write_text(s.replace(OLD_LINE, NEW_LINES), encoding="utf-8")
print("crossmodel_patch.py: save-dir edit inserted after the _reads_ line")

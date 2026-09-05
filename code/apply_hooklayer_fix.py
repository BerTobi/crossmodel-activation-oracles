"""
Fix a definition-order bug in the cross-model patch: `hook_layer = hook_layer_abs` (upstream's `hook_layer = 1` line,
~L946 of sft.py) executed 23 lines BEFORE `hook_layer_abs` was defined in the env block (~L969):
    NameError: name 'hook_layer_abs' is not defined
Now the first assignment reads AO_HOOK_LAYER directly and the late definition becomes a comment.

Usage: python apply_hooklayer_fix.py [repo=activation_oracles] [patch_script=crossmodel_patch.py | /dev/null]
"""
import sys
from pathlib import Path

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else "activation_oracles")
PATCH = Path(sys.argv[2] if len(sys.argv) > 2 else "crossmodel_patch.py")

NL = chr(10)              # real newline (in sft.py)
BSN = chr(92) + "n"       # the two characters backslash + n (how newlines appear inside the patcher's string literals)

OLD_USE = "    hook_layer = hook_layer_abs  # absolute; overridden by hook_onto_layer_percent when that is set" + NL
NEW_USE = ('    hook_layer = int(os.environ.get("AO_HOOK_LAYER", "1"))  # ABSOLUTE oracle injection layer; default 1 (Karvonen). '
           "Overridden by hook_onto_layer_percent when that is set." + NL)
DEF = 'hook_layer_abs: int = int(os.environ.get("AO_HOOK_LAYER", "1"))'
NEW_DEF_BODY = "    #                          (read ABOVE, at `hook_layer = ...`, so it is defined before its first use)"

# ---- repo ----
p = REPO / "nl_probes" / "sft.py"
t = p.read_text(encoding="utf-8")
if OLD_USE not in t and NEW_USE in t:
    print("sft.py already fixed")
else:
    assert t.count(OLD_USE) == 1 and t.count("    " + DEF + NL) == 1, "[sft.py] expected exactly one use and one def"
    t = t.replace(OLD_USE, NEW_USE).replace("    " + DEF + NL, NEW_DEF_BODY + NL)
    p.write_text(t, encoding="utf-8")
    print("applied nl_probes/sft.py (2 edits)")
assert "= hook_layer_abs" not in t, "hook_layer_abs still used"
assert t.index('hook_layer = int(os.environ.get("AO_HOOK_LAYER"') < t.index("hook_onto_layer=hook_layer"), "order"
print("order check OK: hook_layer defined before use")

# ---- patcher (so fresh clones get the fixed file) ----
if not PATCH.exists() or PATCH.name == "null":
    print("no patch script given; repo patched only")
    sys.exit(0)
s = PATCH.read_text(encoding="utf-8")
if DEF not in s and repr(NEW_USE) in s:
    print("crossmodel_patch.py already fixed")
    sys.exit(0)
j = s.index(DEF)
seg_start = s.rindex(BSN, 0, j) + 2     # just after the backslash-n that precedes the def line inside the literal
seg_end = s.index(BSN, j)               # the backslash-n that ends the def line
assert s[seg_start:seg_end].strip() == DEF, f"unexpected patcher segment: {s[seg_start:seg_end]!r}"
s = s[:seg_start] + NEW_DEF_BODY + s[seg_end:]
assert s.count(repr(OLD_USE)) == 1, "[patch] use-edit not found exactly once"
s = s.replace(repr(OLD_USE), repr(NEW_USE))
PATCH.write_text(s, encoding="utf-8")
print("crossmodel_patch.py: hook-layer edits corrected (definition-order fix)")

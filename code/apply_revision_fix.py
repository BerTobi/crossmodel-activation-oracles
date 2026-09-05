"""
Make HF revision pinning PER-MODEL. AO_REVISION pins the SUBJECT (the released oracle's Qwen3-8B base commit);
AO_ORACLE_REVISION (optional) pins the oracle. Previously one `revision` was applied to both models, and a
Qwen3-8B commit hash does not exist in the Qwen3-14B repo -> "Unrecognized model ... model_type" at oracle load.

Also routes the dataset-precompute model_kwargs to the SUBJECT kwargs (dataset building only ever loads the subject).

Usage: python apply_revision_fix.py [repo=activation_oracles] [patch_script=crossmodel_patch.py]
"""
import sys
from pathlib import Path

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else "activation_oracles")
PATCH = Path(sys.argv[2] if len(sys.argv) > 2 else "crossmodel_patch.py")

sft_edits = [
    (
        "        if os.environ.get(\"AO_REVISION\"):\n"
        "            # Pin the HF commit for BOTH oracle and subject (released Qwen3-8B oracle base:\n"
        "            # b968826d9c46dd6066d109eabc6255188de91218). Leave unset when either model is a local path.\n"
        "            model_kwargs[\"revision\"] = os.environ[\"AO_REVISION\"]\n",
        "        # Revisions are PER-MODEL. AO_REVISION pins the SUBJECT (e.g. the released oracle's Qwen3-8B base commit\n"
        "        # b968826d9c46dd6066d109eabc6255188de91218); AO_ORACLE_REVISION pins the oracle. A commit hash from one\n"
        "        # repo does not exist in another (Qwen3-8B hash -> Qwen3-14B load fails with 'Unrecognized model').\n"
        "        subject_model_kwargs = dict(model_kwargs)\n"
        "        if os.environ.get(\"AO_REVISION\"):\n"
        "            subject_model_kwargs[\"revision\"] = os.environ[\"AO_REVISION\"]\n"
        "        if os.environ.get(\"AO_ORACLE_REVISION\"):\n"
        "            model_kwargs[\"revision\"] = os.environ[\"AO_ORACLE_REVISION\"]\n",
        1,
    ),
    (
        "            classification_datasets=classification_datasets,\n"
        "            model_kwargs=model_kwargs,\n",
        "            classification_datasets=classification_datasets,\n"
        "            model_kwargs=subject_model_kwargs,  # dataset precompute loads the SUBJECT\n",
        1,
    ),
    (
        "                model_kwargs=model_kwargs,\n"
        "                verbose=True,\n"
        "            )\n",
        "                model_kwargs=model_kwargs,\n"
        "                subject_model_kwargs=subject_model_kwargs,\n"
        "                verbose=True,\n"
        "            )\n",
        1,
    ),
    (
        "    model_kwargs: dict[str, Any],\n"
        "    verbose: bool = False,\n"
        "):\n",
        "    model_kwargs: dict[str, Any],\n"
        "    verbose: bool = False,\n"
        "    subject_model_kwargs: dict[str, Any] | None = None,\n"
        "):\n",
        1,
    ),
    (
        "        subject_model = load_model(cfg.subject_model_name, dtype, **model_kwargs)\n",
        "        skw = {**(subject_model_kwargs if subject_model_kwargs is not None else model_kwargs),\n"
        "               \"device_map\": {\"\": f\"cuda:{local_rank}\"}}\n"
        "        subject_model = load_model(cfg.subject_model_name, dtype, **skw)\n",
        1,
    ),
]

p = REPO / "nl_probes" / "sft.py"
t = p.read_text(encoding="utf-8")
for old, new, n in sft_edits:
    c = t.count(old)
    assert c == n, f"[sft.py] expected {n}, found {c}: {old[:70]!r}"
    t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print(f"applied nl_probes/sft.py ({len(sft_edits)} edit(s))")

if not PATCH.exists() or PATCH.name == "null":
    print("no patch script given; repo patched only")
    sys.exit(0)
s = PATCH.read_text(encoding="utf-8")
if "AO_ORACLE_REVISION" in s:
    print("crossmodel_patch.py already contains the per-model revision edits; skipping append")
else:
    anchor = "]\nfor old, new, n in edits_sft:\n"
    assert s.count(anchor) == 1
    ins = "    # per-model HF revision (AO_REVISION -> subject only; AO_ORACLE_REVISION -> oracle)\n" + "".join(
        f"    (\n        {old!r},\n        {new!r},\n        {n},\n    ),\n" for old, new, n in sft_edits
    )
    s = s.replace(anchor, ins + anchor)
    PATCH.write_text(s, encoding="utf-8")
    print("crossmodel_patch.py: appended per-model revision edits")

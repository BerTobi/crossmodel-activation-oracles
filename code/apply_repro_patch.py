"""
Two reproducibility fixes from the Misaligned-Oracles handoff docs, applied to a patched checkout
and appended to crossmodel_patch.py.

  (a) get_hf_submodule: fall back to config.model_type so LOCAL checkpoints (e.g. a merged organism at
      runs/M/merged, whose path contains no family name) resolve. Handoff §4b.
  (b) AO_REVISION env -> model_kwargs["revision"] so oracle + subject load a pinned HF commit
      (released oracle base: Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218). Handoff §6.
      Only set it when BOTH models are HF repos (a local path ignores/rejects `revision`).

Usage: python apply_repro_patch.py [repo=activation_oracles] [patch_script=crossmodel_patch.py]
"""
import sys
from pathlib import Path

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else "activation_oracles")
PATCH = Path(sys.argv[2] if len(sys.argv) > 2 else "crossmodel_patch.py")


def apply(rel, edits):
    p = REPO / rel
    t = p.read_text(encoding="utf-8")
    for old, new, n in edits:
        c = t.count(old)
        assert c == n, f"[{rel}] expected {n}, found {c}: {old[:70]!r}"
        t = t.replace(old, new)
    p.write_text(t, encoding="utf-8")
    print(f"applied {rel} ({len(edits)} edit(s))")


au_edits = [
    (
        "    model_name = model.config._name_or_path\n\n    if use_lora:\n",
        "    model_name = model.config._name_or_path\n"
        "    # Local checkpoints (e.g. a merged organism at runs/M/merged) carry no family name in the path;\n"
        "    # fall back to config.model_type (Misaligned-Oracles handoff, section 4b).\n"
        "    model_type = getattr(model.config, \"model_type\", \"\") or \"\"\n"
        "    is_std = (\n"
        "        model_type in (\"qwen3\", \"qwen2\", \"llama\", \"mistral\", \"gemma2\")\n"
        "        or \"gemma-2\" in model_name or \"mistral\" in model_name or \"Llama\" in model_name or \"Qwen\" in model_name\n"
        "    )\n\n"
        "    if use_lora:\n",
        1,
    ),
    (
        "        elif \"gemma-2\" in model_name or \"mistral\" in model_name or \"Llama\" in model_name or \"Qwen\" in model_name:\n"
        "            return model.base_model.model.model.layers[layer]\n",
        "        elif is_std:\n"
        "            return model.base_model.model.model.layers[layer]\n",
        1,
    ),
    (
        "    elif \"gemma-2\" in model_name or \"mistral\" in model_name or \"Llama\" in model_name or \"Qwen\" in model_name:\n"
        "        return model.model.layers[layer]\n",
        "    elif is_std:\n"
        "        return model.model.layers[layer]\n",
        1,
    ),
]

sft_edits = [
    (
        "        if os.environ.get(\"AO_ATTN\"):\n"
        "            model_kwargs[\"attn_implementation\"] = os.environ[\"AO_ATTN\"]\n",
        "        if os.environ.get(\"AO_ATTN\"):\n"
        "            model_kwargs[\"attn_implementation\"] = os.environ[\"AO_ATTN\"]\n"
        "        if os.environ.get(\"AO_REVISION\"):\n"
        "            # Pin the HF commit for BOTH oracle and subject (released Qwen3-8B oracle base:\n"
        "            # b968826d9c46dd6066d109eabc6255188de91218). Leave unset when either model is a local path.\n"
        "            model_kwargs[\"revision\"] = os.environ[\"AO_REVISION\"]\n",
        1,
    ),
]

apply("nl_probes/utils/activation_utils.py", au_edits)
apply("nl_probes/sft.py", sft_edits)

if not PATCH.exists() or PATCH.name == "null":
    print("no patch script given; repo patched only")
    sys.exit(0)
s = PATCH.read_text(encoding="utf-8")
if "7f. Reproducibility" in s:
    print("crossmodel_patch.py already contains the repro section; skipping append")
else:
    marker = "# --------------------------------------------------------------------------------------\n# 8. sft.py:"
    assert s.count(marker) == 1
    body = "".join(f"    (\n        {old!r},\n        {new!r},\n        {n},\n    ),\n" for old, new, n in au_edits)
    rule = "# " + "-" * 86 + "\n"
    block = f"{rule}# 7f. Reproducibility: model_type fallback for local checkpoints (handoff 4b)\n{rule}patch('nl_probes/utils/activation_utils.py', [\n{body}])\n\n"
    s = s.replace(marker, block + marker)
    anchor = "]\nfor old, new, n in edits_sft:\n"
    assert s.count(anchor) == 1
    ins = "    # AO_REVISION: pin the HF commit for oracle + subject\n" + "".join(
        f"    (\n        {old!r},\n        {new!r},\n        {n},\n    ),\n" for old, new, n in sft_edits
    )
    s = s.replace(anchor, ins + anchor)
    PATCH.write_text(s, encoding="utf-8")
    print("crossmodel_patch.py: appended section 7f and the AO_REVISION edit")

"""
Thread `subject_lora_path` into DATASET BUILDING so precomputed activations (classification TEST splits use
save_acts=True) come from the SAME subject the oracle trains on (base + Taboo/organism adapter), not the bare base.

Without this, an oracle trained on clock-organism activations would be evaluated in-loop on clean-Qwen
activations -- a silent train/eval mismatch.

Edits:
  - DatasetLoaderConfig.subject_lora_path (hashes into the cache filename; readable marker added)
  - ClassificationDatasetLoader.create_dataset -> create_vector_dataset(lora_path=subject_lora_path)
  - create_vector_dataset: guard the existing lora_path wrap so it only runs when a model was loaded
  - sft.py: mk_cfg / build_loader_groups take subject_lora_path; every DatasetLoaderConfig gets it

Usage: python apply_subjectlora_patch.py [repo=activation_oracles] [patch_script=crossmodel_patch.py]
"""
import re
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


adm_edits = [
    (
        "    subject_model_name: str | None = None\n\n    @property\n    def effective_subject_model_name(self) -> str:\n",
        "    subject_model_name: str | None = None\n"
        "    # Optional LoRA on the subject (e.g. a Taboo organism). Used when a dataset precomputes activations.\n"
        "    subject_lora_path: str | None = None\n\n"
        "    @property\n    def effective_subject_model_name(self) -> str:\n",
        1,
    ),
    (
        "        if self.dataset_config.subject_model_name is not None:\n"
        "            model_str += \"_reads_\" + self.dataset_config.subject_model_name.split(\"/\")[-1]\n",
        "        if self.dataset_config.subject_model_name is not None:\n"
        "            model_str += \"_reads_\" + self.dataset_config.subject_model_name.split(\"/\")[-1]\n"
        "        if self.dataset_config.subject_lora_path is not None:\n"
        "            model_str += \"_lora_\" + self.dataset_config.subject_lora_path.rstrip(\"/\").split(\"/\")[-1]\n",
        1,
    ),
]

cls_edits = [
    (
        "                model_kwargs=self.model_kwargs,\n"
        "                model=self.model,\n"
        "                subject_tokenizer=subject_tokenizer,\n"
        "            )\n",
        "                model_kwargs=self.model_kwargs,\n"
        "                model=self.model,\n"
        "                lora_path=self.dataset_config.subject_lora_path,  # precomputed acts come from base+adapter\n"
        "                subject_tokenizer=subject_tokenizer,\n"
        "            )\n",
        1,
    ),
    (
        "    if lora_path is not None:\n"
        "        model = PeftModel.from_pretrained(model, lora_path)\n",
        "    if lora_path is not None and model is not None:  # only meaningful when activations are precomputed\n"
        "        model = PeftModel.from_pretrained(model, lora_path)\n"
        "        model.eval()\n",
        1,
    ),
]

apply("nl_probes/dataset_classes/act_dataset_manager.py", adm_edits)
apply("nl_probes/dataset_classes/classification.py", cls_edits)

# ---- sft.py: signatures + companion kwargs (loop, skipping the one call that already has it) ----
p = REPO / "nl_probes" / "sft.py"
t = p.read_text(encoding="utf-8")
sig_old = "    subject_model_name: str | None = None,\n"
assert t.count(sig_old) == 2, t.count(sig_old)  # mk_cfg and build_loader_groups
t = t.replace(sig_old, sig_old + "    subject_lora_path: str | None = None,\n")

lines = t.split("\n")
out, inserted = [], 0
for i, line in enumerate(lines):
    out.append(line)
    m = re.match(r"^(\s+)subject_model_name=subject_model_name,$", line)
    if m:
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if "subject_lora_path" not in nxt:
            out.append(f"{m.group(1)}subject_lora_path=subject_lora_path,")
            inserted += 1
t = "\n".join(out)
assert inserted == 10, f"expected 10 companion insertions, made {inserted}"
p.write_text(t, encoding="utf-8")
print(f"applied nl_probes/sft.py (2 signature edits + {inserted} companion kwargs)")

if not PATCH.exists() or PATCH.name == "null":
    print("no patch script given; repo patched only")
    sys.exit(0)
s = PATCH.read_text(encoding="utf-8")
if "7g. subject_lora_path" in s:
    print("crossmodel_patch.py already contains section 7g; skipping append")
else:
    marker = "# --------------------------------------------------------------------------------------\n# 8. sft.py:"
    assert s.count(marker) == 1
    rule = "# " + "-" * 86 + "\n"

    def block(title, rel, edits):
        body = "".join(f"    (\n        {old!r},\n        {new!r},\n        {n},\n    ),\n" for old, new, n in edits)
        return f"{rule}# {title}\n{rule}patch({rel!r}, [\n{body}])\n\n"

    blocks = (block("7g. subject_lora_path in dataset building (config)", "nl_probes/dataset_classes/act_dataset_manager.py", adm_edits)
              + block("7g. subject_lora_path in dataset building (classification)", "nl_probes/dataset_classes/classification.py", cls_edits))
    s = s.replace(marker, blocks + marker)
    # sft.py part: a code snippet run after edits_sft (signatures + loop), appended near the end
    tail_marker = "sft.write_text(text, encoding=\"utf-8\")\nprint(f\"patched nl_probes/sft.py"
    assert s.count(tail_marker) == 1
    snippet = (
        "# 7g. subject_lora_path: signatures + companion kwargs\n"
        "_sig = \"    subject_model_name: str | None = None,\\n\"\n"
        "assert text.count(_sig) == 2, text.count(_sig)\n"
        "text = text.replace(_sig, _sig + \"    subject_lora_path: str | None = None,\\n\")\n"
        "_lines = text.split(\"\\n\"); _out = []; _ins = 0\n"
        "for _i, _l in enumerate(_lines):\n"
        "    _out.append(_l)\n"
        "    _m = re.match(r\"^(\\s+)subject_model_name=subject_model_name,$\", _l)\n"
        "    if _m and \"subject_lora_path\" not in (_lines[_i + 1] if _i + 1 < len(_lines) else \"\"):\n"
        "        _out.append(f\"{_m.group(1)}subject_lora_path=subject_lora_path,\"); _ins += 1\n"
        "text = \"\\n\".join(_out)\n"
        "assert _ins == 10, _ins\n"
    )
    s = s.replace(tail_marker, snippet + tail_marker)
    PATCH.write_text(s, encoding="utf-8")
    print("crossmodel_patch.py: appended section 7g (config + classification) and the sft.py snippet")

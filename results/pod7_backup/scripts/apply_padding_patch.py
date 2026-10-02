"""
Allow a SUBJECT with smaller hidden_size than the ORACLE by zero-padding captured activations
(lossless). Truncation (subject larger than oracle) stays deliberately excluded: it is lossy and
belongs to a separate arm of the adaptation ladder.

  Qwen3-8B subject (4096)  ->  Qwen3-14B oracle (5120): pad 1024 zeros.

Edits:
  - dataset_utils.materialize_missing_steering_vectors: optional `oracle_hidden_size` (auto-read from the
    oracle when cross_model); pad (K, d_S) -> (K, d_O) when d_S < d_O; error when d_S > d_O.
  - sft.py train_model: replace the equal-size assert with pad-or-fail logic.

Usage: python apply_padding_patch.py [repo=activation_oracles] [patch_script=crossmodel_patch.py]
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


du_edits = [
    (
        "    subject_model: torch.nn.Module | None = None,\n"
        "    subject_tokenizer: AutoTokenizer | None = None,\n"
        ") -> list[TrainingDataPoint]:\n",
        "    subject_model: torch.nn.Module | None = None,\n"
        "    subject_tokenizer: AutoTokenizer | None = None,\n"
        "    oracle_hidden_size: int | None = None,\n"
        ") -> list[TrainingDataPoint]:\n",
        1,
    ),
    (
        "        act_model: torch.nn.Module = subject_model\n"
        "        act_tokenizer = subject_tokenizer\n"
        "    else:\n"
        "        act_model = model\n"
        "        act_tokenizer = tokenizer\n",
        "        act_model: torch.nn.Module = subject_model\n"
        "        act_tokenizer = subject_tokenizer\n"
        "        # Lossless zero-padding when the subject is narrower than the oracle (e.g. Qwen3-8B 4096 -> Qwen3-14B 5120).\n"
        "        if oracle_hidden_size is None:\n"
        "            oracle_hidden_size = int(model.config.hidden_size)\n"
        "    else:\n"
        "        act_model = model\n"
        "        act_tokenizer = tokenizer\n",
        1,
    ),
    (
        "        vectors = acts_BLD[b, idxs, :].detach().contiguous()\n"
        "\n"
        "        assert len(vectors.shape) == 2, f\"Expected 2D tensor, got vectors.shape={vectors.shape}\"\n",
        "        vectors = acts_BLD[b, idxs, :].detach().contiguous()\n"
        "\n"
        "        assert len(vectors.shape) == 2, f\"Expected 2D tensor, got vectors.shape={vectors.shape}\"\n"
        "        if cross_model and oracle_hidden_size is not None and vectors.shape[1] != oracle_hidden_size:\n"
        "            d_s, d_o = vectors.shape[1], oracle_hidden_size\n"
        "            if d_s < d_o:\n"
        "                vectors = torch.nn.functional.pad(vectors, (0, d_o - d_s))  # zero-pad trailing dims (lossless)\n"
        "            else:\n"
        "                raise NotImplementedError(\n"
        "                    f\"Subject hidden_size {d_s} > oracle hidden_size {d_o}: truncation is lossy and deliberately not \"\n"
        "                    \"implemented here (see the adaptation-ladder arm of the plan).\"\n"
        "                )\n",
        1,
    ),
]

sft_edits = [
    (
        "        assert d_oracle == d_subject, (\n"
        "            f\"Adapter-free cross-model requires matching hidden_size; oracle={d_oracle}, subject={d_subject}\"\n"
        "        )\n",
        "        if d_subject < d_oracle:\n"
        "            print(f\"Cross-model: zero-padding subject activations {d_subject} -> {d_oracle} (lossless)\")\n"
        "        elif d_subject > d_oracle:\n"
        "            raise NotImplementedError(\n"
        "                f\"Subject hidden_size {d_subject} > oracle {d_oracle}: truncation is lossy and not implemented\"\n"
        "            )\n",
        1,
    ),
]

apply("nl_probes/utils/dataset_utils.py", du_edits)
apply("nl_probes/sft.py", sft_edits)

if not PATCH.exists() or PATCH.name == "null":
    print("no patch script given; repo patched only")
    sys.exit(0)
s = PATCH.read_text(encoding="utf-8")
if "7e. Zero-padding" in s:
    print("crossmodel_patch.py already contains the padding section; skipping append")
else:
    marker = "# --------------------------------------------------------------------------------------\n# 8. sft.py:"
    assert s.count(marker) == 1
    body = "".join(f"    (\n        {old!r},\n        {new!r},\n        {n},\n    ),\n" for old, new, n in du_edits)
    rule = "# " + "-" * 86 + "\n"
    block = f"{rule}# 7e. Zero-padding for subject hidden_size < oracle hidden_size (lossless; truncation excluded)\n{rule}patch('nl_probes/utils/dataset_utils.py', [\n{body}])\n\n"
    s = s.replace(marker, block + marker)
    anchor = "]\nfor old, new, n in edits_sft:\n"
    assert s.count(anchor) == 1
    ins = "    # zero-padding: replace the equal-hidden-size assert\n" + "".join(
        f"    (\n        {old!r},\n        {new!r},\n        {n},\n    ),\n" for old, new, n in sft_edits
    )
    s = s.replace(anchor, ins + anchor)
    PATCH.write_text(s, encoding="utf-8")
    print("crossmodel_patch.py: appended section 7e and the sft.py padding edit")

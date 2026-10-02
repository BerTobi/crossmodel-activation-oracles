"""
Zero-pad PRECOMPUTED steering vectors too (not only freshly captured ones).

Classification TEST splits store activations captured from the subject at dataset-build time (save_acts=True),
at the SUBJECT's width. materialize_missing_steering_vectors only filled/padded datapoints whose vectors were
None and returned early otherwise, so in eval-on-start a 4096-wide precomputed vector reached the 5120-wide
oracle hook:  "The size of tensor a (4096) must match the size of tensor b (5120)".

Fix: when a subject_model is given (cross-model), walk every datapoint first and pad any present vector that
is narrower than the oracle. Marker in code: [pad_precomputed].

Usage: python apply_padding_fix.py [repo=activation_oracles] [patch_script=crossmodel_patch.py]
"""
import sys
from pathlib import Path

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else "activation_oracles")
PATCH = Path(sys.argv[2] if len(sys.argv) > 2 else "crossmodel_patch.py")

du_edits = [
    (
        "    # Select datapoints that need generation\n"
        "    to_fill: list[tuple[int, TrainingDataPoint]] = [\n",
        "    # [pad_precomputed] Cross-model: vectors that are ALREADY present (e.g. classification test splits built with\n"
        "    # save_acts=True) were captured at the SUBJECT's width; pad them to the ORACLE's width before anything else,\n"
        "    # otherwise the early return below lets a 4096-wide vector reach a 5120-wide hook.\n"
        "    if subject_model is not None:\n"
        "        d_o = int(oracle_hidden_size) if oracle_hidden_size is not None else int(model.config.hidden_size)\n"
        "        repadded: list[TrainingDataPoint] = []\n"
        "        for dp in batch_points:\n"
        "            v = dp.steering_vectors\n"
        "            if v is not None and v.shape[1] != d_o:\n"
        "                if v.shape[1] > d_o:\n"
        "                    raise NotImplementedError(\n"
        "                        f\"Precomputed subject vectors ({v.shape[1]}) wider than oracle ({d_o}): truncation not implemented\"\n"
        "                    )\n"
        "                dp = dp.model_copy(deep=True)\n"
        "                dp.steering_vectors = torch.nn.functional.pad(v, (0, d_o - v.shape[1]))\n"
        "            repadded.append(dp)\n"
        "        batch_points = repadded\n\n"
        "    # Select datapoints that need generation\n"
        "    to_fill: list[tuple[int, TrainingDataPoint]] = [\n",
        1,
    ),
]

p = REPO / "nl_probes" / "utils" / "dataset_utils.py"
t = p.read_text(encoding="utf-8")
for old, new, n in du_edits:
    c = t.count(old)
    assert c == n, f"[dataset_utils.py] expected {n}, found {c}: {old[:60]!r}"
    t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print(f"applied nl_probes/utils/dataset_utils.py ({len(du_edits)} edit(s))")

if not PATCH.exists() or PATCH.name == "null":
    print("no patch script given; repo patched only")
    sys.exit(0)
s = PATCH.read_text(encoding="utf-8")
if "pad_precomputed" in s:
    print("crossmodel_patch.py already contains the pad_precomputed edit; skipping append")
else:
    marker = "# --------------------------------------------------------------------------------------\n# 8. sft.py:"
    assert s.count(marker) == 1
    body = "".join(f"    (\n        {old!r},\n        {new!r},\n        {n},\n    ),\n" for old, new, n in du_edits)
    rule = "# " + "-" * 86 + "\n"
    block = f"{rule}# 7h. Pad PRECOMPUTED subject vectors to the oracle width (eval-on-start crash fix)\n{rule}patch('nl_probes/utils/dataset_utils.py', [\n{body}])\n\n"
    s = s.replace(marker, block + marker)
    PATCH.write_text(s, encoding="utf-8")
    print("crossmodel_patch.py: appended section 7h")

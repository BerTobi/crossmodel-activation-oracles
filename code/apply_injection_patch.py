"""
Add the injection-recipe options to an already cross-model-patched activation_oracles checkout,
and append the same edits to crossmodel_patch.py so fresh clones get them too.

  normalize flag on the hook:   True  -> h' = h + ||h||*v/||v|| * lambda   (Karvonen 2025, default)
                                False -> h' = h + lambda * v              (Bersia & Gaintseva 2026, App. B.3)
  hook_onto_layer_percent:      derive the ORACLE injection layer from its depth (50 -> Qwen3-8B L18, Llama L16)
  env: AO_INJECTION, AO_HOOK_LAYER_PERCENT, AO_LAMBDA

Usage: python apply_injection_patch.py [repo=activation_oracles] [patch_script=crossmodel_patch.py]
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


APOS = chr(39)  # keep the literal apostrophe out of the source to sidestep shell quoting entirely

hooks_edits = [
    (
        "    steering_coefficient: float,\n    device: torch.device,\n    dtype: torch.dtype,\n) -> Callable:\n"
        "    \"\"\"\n    HF hook with debug prints to compare against vLLM.\n",
        "    steering_coefficient: float,\n    device: torch.device,\n    dtype: torch.dtype,\n    normalize: bool = True,\n) -> Callable:\n"
        "    \"\"\"\n    HF hook with debug prints to compare against vLLM.\n\n"
        "    normalize=True  (upstream, Karvonen et al. 2025): resid[pos] += normalize(v) * ||resid[pos]|| * steering_coefficient\n"
        "    normalize=False (raw lambda, Bersia & Gaintseva 2026 App. B.3): resid[pos] += v * steering_coefficient\n",
        1,
    ),
    (
        "    # Pre-normalize once; we never backprop through these\n"
        "    normed_list = [torch.nn.functional.normalize(v_b, dim=-1).detach() for v_b in vectors]\n",
        "    # Pre-normalize once; we never backprop through these\n"
        "    if normalize:\n"
        "        normed_list = [torch.nn.functional.normalize(v_b, dim=-1).detach() for v_b in vectors]\n"
        "    else:\n"
        "        normed_list = [v_b.detach() for v_b in vectors]  # raw: scaled only by steering_coefficient\n",
        1,
    ),
    (
        "            steered_KD = (normed_list[b] *  norms_K1 * steering_coefficient).to(dtype)  # (K_b, d)\n",
        "            if normalize:\n"
        "                steered_KD = (normed_list[b] * norms_K1 * steering_coefficient).to(dtype)  # (K_b, d)\n"
        "            else:\n"
        "                steered_KD = (normed_list[b].to(device) * steering_coefficient).to(dtype)  # (K_b, d) raw lambda\n",
        1,
    ),
]

cfg_edits = [
    (
        "    # Optional LoRA loaded onto the subject (e.g. a Taboo subject for FT-AO training). Kept ACTIVE during capture.\n"
        "    subject_lora_path: str | None = None\n",
        "    # Optional LoRA loaded onto the subject (e.g. a Taboo subject for FT-AO training). Kept ACTIVE during capture.\n"
        "    subject_lora_path: str | None = None\n\n"
        "    # --- Injection recipe ---\n"
        f"    # \"norm_matched\": h{APOS} = h + ||h||*v/||v|| * lambda   (Karvonen et al. 2025, default)\n"
        f"    # \"raw\":          h{APOS} = h + lambda * v              (Bersia & Gaintseva 2026, Appendix B.3; lambda = steering_coefficient)\n"
        "    injection_mode: str = \"norm_matched\"\n"
        "    # If set, hook_onto_layer is derived from the ORACLE depth (50 -> Qwen3-8B L18, Llama-3.1-8B L16).\n"
        "    hook_onto_layer_percent: int | None = None\n",
        1,
    ),
    (
        "        # run name - stable and readable\n",
        "        if self.hook_onto_layer_percent is not None:\n"
        "            self.hook_onto_layer = layer_percent_to_layer(self.model_name, self.hook_onto_layer_percent)\n"
        "        assert self.injection_mode in (\"norm_matched\", \"raw\"), self.injection_mode\n\n"
        "        # run name - stable and readable\n",
        1,
    ),
]

eval_edits = [
    (
        "    steering_coefficient: float,\n    generation_kwargs: dict,\n) -> list[FeatureResult]:\n"
        "    batch_steering_vectors = eval_batch.steering_vectors\n",
        "    steering_coefficient: float,\n    generation_kwargs: dict,\n    normalize_injection: bool = True,\n) -> list[FeatureResult]:\n"
        "    batch_steering_vectors = eval_batch.steering_vectors\n",
        1,
    ),
    (
        "        steering_coefficient=steering_coefficient,\n        device=device,\n        dtype=dtype,\n    )\n\n"
        "    tokenized_input = {\n        \"input_ids\": eval_batch.input_ids,\n",
        "        steering_coefficient=steering_coefficient,\n        device=device,\n        dtype=dtype,\n"
        "        normalize=normalize_injection,\n    )\n\n"
        "    tokenized_input = {\n        \"input_ids\": eval_batch.input_ids,\n",
        1,
    ),
    (
        "    subject_model: torch.nn.Module | None = None,\n    subject_tokenizer: PreTrainedTokenizer | None = None,\n"
        ") -> list[FeatureResult]:\n    \"\"\"Run evaluation and save results.\"\"\"\n",
        "    subject_model: torch.nn.Module | None = None,\n    subject_tokenizer: PreTrainedTokenizer | None = None,\n"
        "    normalize_injection: bool = True,\n"
        ") -> list[FeatureResult]:\n    \"\"\"Run evaluation and save results.\"\"\"\n",
        1,
    ),
    (
        "                steering_coefficient=steering_coefficient,\n                generation_kwargs=generation_kwargs,\n            )\n"
        "            if verbose:\n",
        "                steering_coefficient=steering_coefficient,\n                generation_kwargs=generation_kwargs,\n"
        "                normalize_injection=normalize_injection,\n            )\n"
        "            if verbose:\n",
        1,
    ),
]

sft_edits = [
    (
        "        steering_coefficient=cfg.steering_coefficient,\n        device=device,\n        dtype=dtype,\n    )\n\n"
        "    tokenized_input = {\n        \"input_ids\": training_batch.input_ids,\n",
        "        steering_coefficient=cfg.steering_coefficient,\n        device=device,\n        dtype=dtype,\n"
        "        normalize=cfg.injection_mode == \"norm_matched\",\n    )\n\n"
        "    tokenized_input = {\n        \"input_ids\": training_batch.input_ids,\n",
        1,
    ),
    (
        "            subject_model=subject_model,\n            subject_tokenizer=subject_tokenizer,\n        )\n"
        "        percent_format_correct, percent_ans_correct = score_eval_responses(eval_responses, eval_datasets[ds])\n",
        "            subject_model=subject_model,\n            subject_tokenizer=subject_tokenizer,\n"
        "            normalize_injection=cfg.injection_mode == \"norm_matched\",\n        )\n"
        "        percent_format_correct, percent_ans_correct = score_eval_responses(eval_responses, eval_datasets[ds])\n",
        1,
    ),
    (
        "    submodule = get_hf_submodule(model, cfg.hook_onto_layer)\n\n    if cfg.use_lora and cfg.load_lora_path is None:\n",
        "    submodule = get_hf_submodule(model, cfg.hook_onto_layer)\n"
        "    print(f\"Injection: mode={cfg.injection_mode} lambda={cfg.steering_coefficient} at ORACLE layer {cfg.hook_onto_layer}\")\n\n"
        "    if cfg.use_lora and cfg.load_lora_path is None:\n",
        1,
    ),
    (
        "    subject_lora_path: str | None = os.environ.get(\"AO_SUBJECT_LORA\", \"\") or None\n",
        "    subject_lora_path: str | None = os.environ.get(\"AO_SUBJECT_LORA\", \"\") or None\n"
        "    #   AO_INJECTION         : norm_matched (Karvonen default) | raw (Bersia & Gaintseva App. B.3)\n"
        "    #   AO_HOOK_LAYER_PERCENT: e.g. 50 -> inject at 50% of the ORACLE depth (Qwen3-8B L18, Llama L16); unset -> layer 1\n"
        "    #   AO_LAMBDA            : steering coefficient (default 1.0)\n"
        "    injection_mode: str = os.environ.get(\"AO_INJECTION\", \"norm_matched\")\n"
        "    hook_layer_percent: int | None = (\n"
        "        int(os.environ[\"AO_HOOK_LAYER_PERCENT\"]) if os.environ.get(\"AO_HOOK_LAYER_PERCENT\") else None\n"
        "    )\n"
        "    steering_lambda: float = float(os.environ.get(\"AO_LAMBDA\", \"1.0\"))\n",
        1,
    ),
    (
        "                subject_lora_path=subject_lora_path,\n                hook_onto_layer=hook_layer,\n",
        "                subject_lora_path=subject_lora_path,\n                hook_onto_layer=hook_layer,\n"
        "                hook_onto_layer_percent=hook_layer_percent,\n"
        "                injection_mode=injection_mode,\n"
        "                steering_coefficient=steering_lambda,\n",
        1,
    ),
]

apply("nl_probes/utils/steering_hooks.py", hooks_edits)
apply("nl_probes/configs/sft_config.py", cfg_edits)
apply("nl_probes/utils/eval.py", eval_edits)
apply("nl_probes/sft.py", sft_edits)

# ---- append the same edits to the reproducible patch script (idempotent) ----
if not PATCH.exists() or PATCH.name == "null":
    print("no patch script given; repo patched only"); sys.exit(0)
s = PATCH.read_text(encoding="utf-8")
if "7b. Injection recipe" in s:
    print("crossmodel_patch.py already contains the injection sections; skipping append")
else:
    marker = "# --------------------------------------------------------------------------------------\n# 8. sft.py:"
    assert s.count(marker) == 1

    def block(title, rel, edits):
        body = "".join(f"    (\n        {old!r},\n        {new!r},\n        {n},\n    ),\n" for old, new, n in edits)
        rule = "# " + "-" * 86 + "\n"
        return f"{rule}# {title}\n{rule}patch({rel!r}, [\n{body}])\n\n"

    new_blocks = (
        block("7b. Injection recipe: normalize flag on the hook (raw-lambda mode, Bersia & Gaintseva App. B.3)", "nl_probes/utils/steering_hooks.py", hooks_edits)
        + block("7c. Config: injection_mode + hook_onto_layer_percent", "nl_probes/configs/sft_config.py", cfg_edits)
        + block("7d. eval.py: thread normalize_injection through", "nl_probes/utils/eval.py", eval_edits)
    )
    s = s.replace(marker, new_blocks + marker)
    anchor = "]\nfor old, new, n in edits_sft:\n"
    assert s.count(anchor) == 1
    ins = "    # injection recipe plumbing (normalize flag, hook layer by percent, env vars)\n" + "".join(
        f"    (\n        {old!r},\n        {new!r},\n        {n},\n    ),\n" for old, new, n in sft_edits
    )
    s = s.replace(anchor, ins + anchor)
    PATCH.write_text(s, encoding="utf-8")
    print("crossmodel_patch.py: appended sections 7b-7d and the sft.py injection edits")

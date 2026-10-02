"""
Apply the cross-model patch to a fresh clone of adamkarvonen/activation_oracles.

Design: add an optional `subject_model_name` everywhere. When it is None, behaviour is
byte-identical to upstream (so Karvonen's released self-oracles remain valid anchors).
When set, the ORACLE prompt is tokenized with the oracle's tokenizer and the CONTEXT the
subject reads is tokenized with the subject's tokenizer; activations are captured from a
separate, frozen subject model instead of the oracle's own base weights.

Every replacement asserts the target string occurs exactly the expected number of times,
so upstream drift fails loudly rather than mis-patching.

Usage:  python crossmodel_patch.py [path/to/activation_oracles]
"""

import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "activation_oracles").resolve()
assert (ROOT / "nl_probes" / "sft.py").exists(), f"Not an activation_oracles checkout: {ROOT}"


def patch(rel: str, edits: list[tuple[str, str, int]]) -> None:
    """edits: (old, new, expected_count). expected_count=-1 means 'at least one, replace all'."""
    p = ROOT / rel
    text = p.read_text(encoding="utf-8")
    for old, new, n in edits:
        c = text.count(old)
        if n == -1:
            assert c >= 1, f"[{rel}] expected >=1 occurrence, found {c}:\n{old!r}"
        else:
            assert c == n, f"[{rel}] expected {n} occurrence(s), found {c}:\n{old!r}"
        text = text.replace(old, new)
    p.write_text(text, encoding="utf-8")
    print(f"patched {rel}  ({len(edits)} edit(s))")


# --------------------------------------------------------------------------------------
# 1. DatasetLoaderConfig: add subject_model_name (hashes into the cache filename automatically)
# --------------------------------------------------------------------------------------
patch("nl_probes/dataset_classes/act_dataset_manager.py", [
    (
        "    dataset_name: str = \"\"\n"
        "    dataset_folder: str = \"sft_training_data\"\n"
        "    seed: int = 42\n",
        "    dataset_name: str = \"\"\n"
        "    dataset_folder: str = \"sft_training_data\"\n"
        "    seed: int = 42\n"
        "    # Cross-model: the model whose activations are read. None -> same as model_name (upstream behaviour).\n"
        "    # The oracle prompt is tokenized with model_name's tokenizer; the context with subject_model_name's.\n"
        "    subject_model_name: str | None = None\n"
        "\n"
        "    @property\n"
        "    def effective_subject_model_name(self) -> str:\n"
        "        return self.subject_model_name or self.model_name\n",
        1,
    ),
    (
        "        model_str = self.dataset_config.model_name.split(\"/\")[-1]\n",
        "        model_str = self.dataset_config.model_name.split(\"/\")[-1]\n"
        "        if self.dataset_config.subject_model_name is not None:\n"
        "            model_str += \"_reads_\" + self.dataset_config.subject_model_name.split(\"/\")[-1]\n",
        1,
    ),
])

# --------------------------------------------------------------------------------------
# 2. Past-lens dataset: context rendered/tokenized/decoded with the SUBJECT tokenizer
# --------------------------------------------------------------------------------------
patch("nl_probes/dataset_classes/past_lens_dataset.py", [
    (
        "    def create_dataset(self) -> None:\n"
        "        tokenizer = load_tokenizer(self.dataset_config.model_name)\n"
        "        dataset = hf_mixed_dataset_to_generator(tokenizer)\n",
        "    def create_dataset(self) -> None:\n"
        "        tokenizer = load_tokenizer(self.dataset_config.model_name)\n"
        "        subject_tokenizer = load_tokenizer(self.dataset_config.effective_subject_model_name)\n"
        "        # Context text is rendered with the SUBJECT's chat template / special tokens: the subject reads it.\n"
        "        dataset = hf_mixed_dataset_to_generator(subject_tokenizer)\n",
        1,
    ),
    (
        "            num_datapoints=self.dataset_config.num_train,\n"
        "            dtype=dtype,\n"
        "        )\n",
        "            num_datapoints=self.dataset_config.num_train,\n"
        "            dtype=dtype,\n"
        "            subject_tokenizer=subject_tokenizer,\n"
        "        )\n",
        1,
    ),
    (
        "    num_datapoints: int,\n"
        "    dtype: torch.dtype,\n"
        ") -> list[TrainingDataPoint]:\n"
        "    random.seed(dataset_config.seed)\n"
        "    torch.manual_seed(dataset_config.seed)\n"
        "\n"
        "    layers = [\n"
        "        layer_percent_to_layer(dataset_config.model_name, layer_percent)\n"
        "        for layer_percent in dataset_config.layer_percents\n"
        "    ]\n"
        "\n"
        "    device = torch.device(\"cpu\")\n"
        "    if dataset_config.save_acts:\n"
        "        model = load_model(dataset_config.model_name, dtype)\n",
        "    num_datapoints: int,\n"
        "    dtype: torch.dtype,\n"
        "    subject_tokenizer: AutoTokenizer | None = None,\n"
        ") -> list[TrainingDataPoint]:\n"
        "    # `tokenizer` builds the ORACLE prompt; `subject_tokenizer` tokenizes the context the SUBJECT reads.\n"
        "    if subject_tokenizer is None:\n"
        "        subject_tokenizer = tokenizer\n"
        "    subject_model_name = dataset_config.effective_subject_model_name\n"
        "\n"
        "    random.seed(dataset_config.seed)\n"
        "    torch.manual_seed(dataset_config.seed)\n"
        "\n"
        "    layers = [\n"
        "        layer_percent_to_layer(subject_model_name, layer_percent)\n"
        "        for layer_percent in dataset_config.layer_percents\n"
        "    ]\n"
        "\n"
        "    device = torch.device(\"cpu\")\n"
        "    if dataset_config.save_acts:\n"
        "        model = load_model(subject_model_name, dtype)\n",
        1,
    ),
    (
        "        tokenized_inputs = tokenizer(\n",
        "        tokenized_inputs = subject_tokenizer(\n",
        1,
    ),
    (
        "                    target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)\n",
        "                    target_text = subject_tokenizer.decode(target_tokens, skip_special_tokens=True)\n",
        2,
    ),
])

# --------------------------------------------------------------------------------------
# 3. Classification dataset
# --------------------------------------------------------------------------------------
patch("nl_probes/dataset_classes/classification.py", [
    (
        "        self.act_layers = [\n"
        "            layer_percent_to_layer(self.dataset_config.model_name, layer_percent)\n",
        "        self.act_layers = [\n"
        "            layer_percent_to_layer(self.dataset_config.effective_subject_model_name, layer_percent)\n",
        1,
    ),
    (
        "    def create_dataset(self) -> None:\n"
        "        tokenizer = load_tokenizer(self.dataset_config.model_name)\n"
        "\n"
        "        train_datapoints, test_datapoints = get_classification_datapoints(\n",
        "    def create_dataset(self) -> None:\n"
        "        tokenizer = load_tokenizer(self.dataset_config.model_name)\n"
        "        subject_tokenizer = load_tokenizer(self.dataset_config.effective_subject_model_name)\n"
        "\n"
        "        train_datapoints, test_datapoints = get_classification_datapoints(\n",
        1,
    ),
    (
        "            data = create_vector_dataset(\n"
        "                datapoints,\n"
        "                tokenizer,\n"
        "                self.dataset_config.model_name,\n",
        "            data = create_vector_dataset(\n"
        "                datapoints,\n"
        "                tokenizer,\n"
        "                self.dataset_config.effective_subject_model_name,  # model loaded for save_acts is the SUBJECT\n",
        1,
    ),
    (
        "                model_kwargs=self.model_kwargs,\n"
        "                model=self.model,\n"
        "            )\n",
        "                model_kwargs=self.model_kwargs,\n"
        "                model=self.model,\n"
        "                subject_tokenizer=subject_tokenizer,\n"
        "            )\n",
        1,
    ),
    (
        "    model_kwargs: dict[str, Any] | None = None,\n"
        "    model=None,\n"
        ") -> list[TrainingDataPoint]:\n"
        "    assert min_end_offset < 0, \"Min end offset must be negative\"\n",
        "    model_kwargs: dict[str, Any] | None = None,\n"
        "    model=None,\n"
        "    subject_tokenizer: AutoTokenizer | None = None,\n"
        ") -> list[TrainingDataPoint]:\n"
        "    # `tokenizer` builds the ORACLE prompt; `subject_tokenizer` tokenizes the context the SUBJECT reads.\n"
        "    # `model_name` / `model` refer to the SUBJECT (only used when save_acts=True).\n"
        "    if subject_tokenizer is None:\n"
        "        subject_tokenizer = tokenizer\n"
        "    assert min_end_offset < 0, \"Min end offset must be negative\"\n",
        1,
    ),
    (
        "    assert tokenizer.padding_side == \"left\", \"Padding side must be left\"\n",
        "    assert tokenizer.padding_side == \"left\", \"Padding side must be left\"\n"
        "    assert subject_tokenizer.padding_side == \"left\", \"Subject padding side must be left\"\n",
        1,
    ),
    (
        "        tokenized_prompts = tokenizer.apply_chat_template(formatted_prompts, tokenize=False)\n"
        "        tokenized_prompts = tokenizer(\n",
        "        tokenized_prompts = subject_tokenizer.apply_chat_template(formatted_prompts, tokenize=False)\n"
        "        tokenized_prompts = subject_tokenizer(\n",
        1,
    ),
    (
        "                    view_tokens(input_ids_L, tokenizer, positions_K[-1])\n",
        "                    view_tokens(input_ids_L, subject_tokenizer, positions_K[-1])\n",
        1,
    ),
])

# --------------------------------------------------------------------------------------
# 4. LatentQA dataset
# --------------------------------------------------------------------------------------
patch("nl_probes/dataset_classes/latentqa_dataset.py", [
    (
        "    def create_dataset(self) -> None:\n"
        "        tokenizer = load_tokenizer(self.dataset_config.model_name)\n"
        "\n"
        "        layers = [\n"
        "            layer_percent_to_layer(self.dataset_config.model_name, layer_percent)\n",
        "    def create_dataset(self) -> None:\n"
        "        tokenizer = load_tokenizer(self.dataset_config.model_name)\n"
        "        subject_tokenizer = load_tokenizer(self.dataset_config.effective_subject_model_name)\n"
        "\n"
        "        layers = [\n"
        "            layer_percent_to_layer(self.dataset_config.effective_subject_model_name, layer_percent)\n",
        1,
    ),
    (
        "            training_data.append(create_latentqa_training_datapoint(dp, tokenizer, layers, self.dataset_params))\n",
        "            training_data.append(\n"
        "                create_latentqa_training_datapoint(dp, tokenizer, layers, self.dataset_params, subject_tokenizer)\n"
        "            )\n",
        1,
    ),
    (
        "def create_latentqa_training_datapoint(\n"
        "    datapoint_dict: dict, tokenizer: AutoTokenizer, act_layers: list[int], dataset_params: LatentQADatasetConfig\n"
        ") -> TrainingDataPoint:\n"
        "    masked_turn_count = {\"stimulus_completion\": 2, \"stimulus\": 2, \"control\": 0}\n",
        "def create_latentqa_training_datapoint(\n"
        "    datapoint_dict: dict,\n"
        "    tokenizer: AutoTokenizer,\n"
        "    act_layers: list[int],\n"
        "    dataset_params: LatentQADatasetConfig,\n"
        "    subject_tokenizer: AutoTokenizer | None = None,\n"
        ") -> TrainingDataPoint:\n"
        "    # `tokenizer` builds the ORACLE prompt; `subject_tokenizer` tokenizes the context the SUBJECT reads.\n"
        "    if subject_tokenizer is None:\n"
        "        subject_tokenizer = tokenizer\n"
        "    masked_turn_count = {\"stimulus_completion\": 2, \"stimulus\": 2, \"control\": 0}\n",
        1,
    ),
    (
        "        masked_str = tokenizer.apply_chat_template(masked_turns, tokenize=False, enable_thinking=False)\n"
        "        masked_tokens = tokenizer(masked_str, return_tensors=None, add_special_tokens=False, padding=False)[\"input_ids\"]\n",
        "        masked_str = subject_tokenizer.apply_chat_template(masked_turns, tokenize=False, enable_thinking=False)\n"
        "        masked_tokens = subject_tokenizer(masked_str, return_tensors=None, add_special_tokens=False, padding=False)[\n"
        "            \"input_ids\"\n"
        "        ]\n",
        1,
    ),
    (
        "    full_read_str = tokenizer.apply_chat_template(\n",
        "    full_read_str = subject_tokenizer.apply_chat_template(\n",
        1,
    ),
    (
        "    context_input_ids = tokenizer(full_read_str, return_tensors=None, add_special_tokens=False, padding=False)[\n",
        "    context_input_ids = subject_tokenizer(full_read_str, return_tensors=None, add_special_tokens=False, padding=False)[\n",
        1,
    ),
])

# --------------------------------------------------------------------------------------
# 5. materialize_missing_steering_vectors: capture from a separate subject model
# --------------------------------------------------------------------------------------
patch("nl_probes/utils/dataset_utils.py", [
    (
        "def materialize_missing_steering_vectors(\n"
        "    batch_points: list[TrainingDataPoint],\n"
        "    tokenizer: AutoTokenizer,\n"
        "    model: PeftModel,\n"
        ") -> list[TrainingDataPoint]:\n",
        "def materialize_missing_steering_vectors(\n"
        "    batch_points: list[TrainingDataPoint],\n"
        "    tokenizer: AutoTokenizer,\n"
        "    model: PeftModel,\n"
        "    subject_model: torch.nn.Module | None = None,\n"
        "    subject_tokenizer: AutoTokenizer | None = None,\n"
        ") -> list[TrainingDataPoint]:\n",
        1,
    ),
    (
        "    assert isinstance(model, PeftModel), \"Model must be a PeftModel\"\n"
        "\n"
        "    # Validate context fields\n",
        "    assert isinstance(model, PeftModel), \"Model must be a PeftModel\"\n"
        "\n"
        "    # Cross-model: read activations from a separate subject model instead of the oracle's own base weights.\n"
        "    cross_model = subject_model is not None\n"
        "    if cross_model:\n"
        "        assert subject_tokenizer is not None, \"subject_tokenizer is required when subject_model is given\"\n"
        "        act_model: torch.nn.Module = subject_model\n"
        "        act_tokenizer = subject_tokenizer\n"
        "    else:\n"
        "        act_model = model\n"
        "        act_tokenizer = tokenizer\n"
        "\n"
        "    # Validate context fields\n",
        1,
    ),
    (
        "    pad_id = tokenizer.pad_token_id\n",
        "    pad_id = act_tokenizer.pad_token_id\n",
        1,
    ),
    (
        "    device = next(model.parameters()).device\n",
        "    device = next(act_model.parameters()).device\n",
        1,
    ),
    (
        "    submodules = {layer: get_hf_submodule(model, layer, use_lora=True) for layer in layers_needed}\n"
        "\n"
        "    # Run a single pass with dropout off, then restore the previous train/eval mode\n"
        "    was_training = model.training\n"
        "    model.eval()\n"
        "    with model.disable_adapter():\n"
        "        # [layer] -> [B, L, D], where B == len(to_fill)\n"
        "        acts_by_layer = collect_activations_multiple_layers(\n"
        "            model=model,\n"
        "            submodules=submodules,\n"
        "            inputs_BL=inputs_BL,\n"
        "            min_offset=None,\n"
        "            max_offset=None,\n"
        "        )\n"
        "    if was_training:\n"
        "        model.train()\n",
        "    submodules = {\n"
        "        layer: get_hf_submodule(act_model, layer, use_lora=isinstance(act_model, PeftModel)) for layer in layers_needed\n"
        "    }\n"
        "\n"
        "    # Run a single pass with dropout off, then restore the previous train/eval mode\n"
        "    was_training = act_model.training\n"
        "    act_model.eval()\n"
        "    if cross_model:\n"
        "        # Separate subject. If it carries a LoRA (e.g. a Taboo subject) keep it ACTIVE: that adapter *is*\n"
        "        # the subject we want to read. Frozen, so no_grad is safe and saves memory.\n"
        "        with torch.no_grad():\n"
        "            acts_by_layer = collect_activations_multiple_layers(\n"
        "                model=act_model,\n"
        "                submodules=submodules,\n"
        "                inputs_BL=inputs_BL,\n"
        "                min_offset=None,\n"
        "                max_offset=None,\n"
        "            )\n"
        "    else:\n"
        "        with model.disable_adapter():\n"
        "            # [layer] -> [B, L, D], where B == len(to_fill)\n"
        "            acts_by_layer = collect_activations_multiple_layers(\n"
        "                model=model,\n"
        "                submodules=submodules,\n"
        "                inputs_BL=inputs_BL,\n"
        "                min_offset=None,\n"
        "                max_offset=None,\n"
        "            )\n"
        "    if was_training:\n"
        "        act_model.train()\n",
        1,
    ),
])

# --------------------------------------------------------------------------------------
# 6. run_evaluation: thread the subject through
# --------------------------------------------------------------------------------------
patch("nl_probes/utils/eval.py", [
    (
        "    generation_kwargs: dict,\n"
        "    verbose: bool = False,\n"
        ") -> list[FeatureResult]:\n"
        "    \"\"\"Run evaluation and save results.\"\"\"\n",
        "    generation_kwargs: dict,\n"
        "    verbose: bool = False,\n"
        "    subject_model: torch.nn.Module | None = None,\n"
        "    subject_tokenizer: PreTrainedTokenizer | None = None,\n"
        ") -> list[FeatureResult]:\n"
        "    \"\"\"Run evaluation and save results.\"\"\"\n",
        1,
    ),
    (
        "            e_batch = materialize_missing_steering_vectors(e_batch, tokenizer, model)\n",
        "            e_batch = materialize_missing_steering_vectors(\n"
        "                e_batch, tokenizer, model, subject_model=subject_model, subject_tokenizer=subject_tokenizer\n"
        "            )\n",
        1,
    ),
])

# --------------------------------------------------------------------------------------
# 7. Training config
# --------------------------------------------------------------------------------------
patch("nl_probes/configs/sft_config.py", [
    (
        "    # --- Model ---\n"
        "    model_name: str = \"Qwen/Qwen3-8B\"\n"
        "    hook_onto_layer: int = 1\n"
        "    layer_percents: list[int] = field(default_factory=lambda: [25, 50, 75])\n"
        "    act_layers: list[int] = field(default_factory=list)  # derived if empty\n",
        "    # --- Model ---\n"
        "    model_name: str = \"Qwen/Qwen3-8B\"  # the ORACLE backbone (carries the LoRA)\n"
        "    hook_onto_layer: int = 1  # ORACLE layer at which subject activations are injected\n"
        "    layer_percents: list[int] = field(default_factory=lambda: [25, 50, 75])\n"
        "    act_layers: list[int] = field(default_factory=list)  # derived if empty, from the SUBJECT's depth\n"
        "\n"
        "    # --- Cross-model (optional) ---\n"
        "    # Model whose activations the oracle reads. None -> oracle reads its own base weights (upstream recipe).\n"
        "    subject_model_name: str | None = None\n"
        "    # Optional LoRA loaded onto the subject (e.g. a Taboo subject for FT-AO training). Kept ACTIVE during capture.\n"
        "    subject_lora_path: str | None = None\n"
        "\n"
        "    @property\n"
        "    def effective_subject_model_name(self) -> str:\n"
        "        return self.subject_model_name or self.model_name\n",
        1,
    ),
    (
        "            self.act_layers = [layer_percent_to_layer(self.model_name, p) for p in self.layer_percents]\n",
        "            self.act_layers = [\n"
        "                layer_percent_to_layer(self.effective_subject_model_name, p) for p in self.layer_percents\n"
        "            ]\n",
        1,
    ),
])

# --------------------------------------------------------------------------------------
# 7b. Injection recipe: normalize flag on the hook (raw-lambda mode, Bersia & Gaintseva App. B.3)
# --------------------------------------------------------------------------------------
patch('nl_probes/utils/steering_hooks.py', [
    (
        '    steering_coefficient: float,\n    device: torch.device,\n    dtype: torch.dtype,\n) -> Callable:\n    """\n    HF hook with debug prints to compare against vLLM.\n',
        '    steering_coefficient: float,\n    device: torch.device,\n    dtype: torch.dtype,\n    normalize: bool = True,\n) -> Callable:\n    """\n    HF hook with debug prints to compare against vLLM.\n\n    normalize=True  (upstream, Karvonen et al. 2025): resid[pos] += normalize(v) * ||resid[pos]|| * steering_coefficient\n    normalize=False (raw lambda, Bersia & Gaintseva 2026 App. B.3): resid[pos] += v * steering_coefficient\n',
        1,
    ),
    (
        '    # Pre-normalize once; we never backprop through these\n    normed_list = [torch.nn.functional.normalize(v_b, dim=-1).detach() for v_b in vectors]\n',
        '    # Pre-normalize once; we never backprop through these\n    if normalize:\n        normed_list = [torch.nn.functional.normalize(v_b, dim=-1).detach() for v_b in vectors]\n    else:\n        normed_list = [v_b.detach() for v_b in vectors]  # raw: scaled only by steering_coefficient\n',
        1,
    ),
    (
        '            steered_KD = (normed_list[b] *  norms_K1 * steering_coefficient).to(dtype)  # (K_b, d)\n',
        '            if normalize:\n                steered_KD = (normed_list[b] * norms_K1 * steering_coefficient).to(dtype)  # (K_b, d)\n            else:\n                steered_KD = (normed_list[b].to(device) * steering_coefficient).to(dtype)  # (K_b, d) raw lambda\n',
        1,
    ),
])

# --------------------------------------------------------------------------------------
# 7c. Config: injection_mode + hook_onto_layer_percent
# --------------------------------------------------------------------------------------
patch('nl_probes/configs/sft_config.py', [
    (
        '    # Optional LoRA loaded onto the subject (e.g. a Taboo subject for FT-AO training). Kept ACTIVE during capture.\n    subject_lora_path: str | None = None\n',
        '    # Optional LoRA loaded onto the subject (e.g. a Taboo subject for FT-AO training). Kept ACTIVE during capture.\n    subject_lora_path: str | None = None\n\n    # --- Injection recipe ---\n    # "norm_matched": h\' = h + ||h||*v/||v|| * lambda   (Karvonen et al. 2025, default)\n    # "raw":          h\' = h + lambda * v              (Bersia & Gaintseva 2026, Appendix B.3; lambda = steering_coefficient)\n    injection_mode: str = "norm_matched"\n    # If set, hook_onto_layer is derived from the ORACLE depth (50 -> Qwen3-8B L18, Llama-3.1-8B L16).\n    hook_onto_layer_percent: int | None = None\n',
        1,
    ),
    (
        '        # run name - stable and readable\n',
        '        if self.hook_onto_layer_percent is not None:\n            self.hook_onto_layer = layer_percent_to_layer(self.model_name, self.hook_onto_layer_percent)\n        assert self.injection_mode in ("norm_matched", "raw"), self.injection_mode\n\n        # run name - stable and readable\n',
        1,
    ),
])

# --------------------------------------------------------------------------------------
# 7d. eval.py: thread normalize_injection through
# --------------------------------------------------------------------------------------
patch('nl_probes/utils/eval.py', [
    (
        '    steering_coefficient: float,\n    generation_kwargs: dict,\n) -> list[FeatureResult]:\n    batch_steering_vectors = eval_batch.steering_vectors\n',
        '    steering_coefficient: float,\n    generation_kwargs: dict,\n    normalize_injection: bool = True,\n) -> list[FeatureResult]:\n    batch_steering_vectors = eval_batch.steering_vectors\n',
        1,
    ),
    (
        '        steering_coefficient=steering_coefficient,\n        device=device,\n        dtype=dtype,\n    )\n\n    tokenized_input = {\n        "input_ids": eval_batch.input_ids,\n',
        '        steering_coefficient=steering_coefficient,\n        device=device,\n        dtype=dtype,\n        normalize=normalize_injection,\n    )\n\n    tokenized_input = {\n        "input_ids": eval_batch.input_ids,\n',
        1,
    ),
    (
        '    subject_model: torch.nn.Module | None = None,\n    subject_tokenizer: PreTrainedTokenizer | None = None,\n) -> list[FeatureResult]:\n    """Run evaluation and save results."""\n',
        '    subject_model: torch.nn.Module | None = None,\n    subject_tokenizer: PreTrainedTokenizer | None = None,\n    normalize_injection: bool = True,\n) -> list[FeatureResult]:\n    """Run evaluation and save results."""\n',
        1,
    ),
    (
        '                steering_coefficient=steering_coefficient,\n                generation_kwargs=generation_kwargs,\n            )\n            if verbose:\n',
        '                steering_coefficient=steering_coefficient,\n                generation_kwargs=generation_kwargs,\n                normalize_injection=normalize_injection,\n            )\n            if verbose:\n',
        1,
    ),
])

# --------------------------------------------------------------------------------------
# 7e. Zero-padding for subject hidden_size < oracle hidden_size (lossless; truncation excluded)
# --------------------------------------------------------------------------------------
patch('nl_probes/utils/dataset_utils.py', [
    (
        '    subject_model: torch.nn.Module | None = None,\n    subject_tokenizer: AutoTokenizer | None = None,\n) -> list[TrainingDataPoint]:\n',
        '    subject_model: torch.nn.Module | None = None,\n    subject_tokenizer: AutoTokenizer | None = None,\n    oracle_hidden_size: int | None = None,\n) -> list[TrainingDataPoint]:\n',
        1,
    ),
    (
        '        act_model: torch.nn.Module = subject_model\n        act_tokenizer = subject_tokenizer\n    else:\n        act_model = model\n        act_tokenizer = tokenizer\n',
        '        act_model: torch.nn.Module = subject_model\n        act_tokenizer = subject_tokenizer\n        # Lossless zero-padding when the subject is narrower than the oracle (e.g. Qwen3-8B 4096 -> Qwen3-14B 5120).\n        if oracle_hidden_size is None:\n            oracle_hidden_size = int(model.config.hidden_size)\n    else:\n        act_model = model\n        act_tokenizer = tokenizer\n',
        1,
    ),
    (
        '        vectors = acts_BLD[b, idxs, :].detach().contiguous()\n\n        assert len(vectors.shape) == 2, f"Expected 2D tensor, got vectors.shape={vectors.shape}"\n',
        '        vectors = acts_BLD[b, idxs, :].detach().contiguous()\n\n        assert len(vectors.shape) == 2, f"Expected 2D tensor, got vectors.shape={vectors.shape}"\n        if cross_model and oracle_hidden_size is not None and vectors.shape[1] != oracle_hidden_size:\n            d_s, d_o = vectors.shape[1], oracle_hidden_size\n            if d_s < d_o:\n                vectors = torch.nn.functional.pad(vectors, (0, d_o - d_s))  # zero-pad trailing dims (lossless)\n            else:\n                vectors = vectors[:, :d_o].contiguous()  # [truncate] keep the first d_o dims (lossy; adaptation-ladder arm)\n',
        1,
    ),
])

# --------------------------------------------------------------------------------------
# 7f. Reproducibility: model_type fallback for local checkpoints (handoff 4b)
# --------------------------------------------------------------------------------------
patch('nl_probes/utils/activation_utils.py', [
    (
        '    model_name = model.config._name_or_path\n\n    if use_lora:\n',
        '    model_name = model.config._name_or_path\n    # Local checkpoints (e.g. a merged organism at runs/M/merged) carry no family name in the path;\n    # fall back to config.model_type (Misaligned-Oracles handoff, section 4b).\n    model_type = getattr(model.config, "model_type", "") or ""\n    is_std = (\n        model_type in ("qwen3", "qwen2", "llama", "mistral", "gemma2")\n        or "gemma-2" in model_name or "mistral" in model_name or "Llama" in model_name or "Qwen" in model_name\n    )\n\n    if use_lora:\n',
        1,
    ),
    (
        '        elif "gemma-2" in model_name or "mistral" in model_name or "Llama" in model_name or "Qwen" in model_name:\n            return model.base_model.model.model.layers[layer]\n',
        '        elif is_std:\n            return model.base_model.model.model.layers[layer]\n',
        1,
    ),
    (
        '    elif "gemma-2" in model_name or "mistral" in model_name or "Llama" in model_name or "Qwen" in model_name:\n        return model.model.layers[layer]\n',
        '    elif is_std:\n        return model.model.layers[layer]\n',
        1,
    ),
])

# --------------------------------------------------------------------------------------
# 7g. subject_lora_path in dataset building (config)
# --------------------------------------------------------------------------------------
patch('nl_probes/dataset_classes/act_dataset_manager.py', [
    (
        '    subject_model_name: str | None = None\n\n    @property\n    def effective_subject_model_name(self) -> str:\n',
        '    subject_model_name: str | None = None\n    # Optional LoRA on the subject (e.g. a Taboo organism). Used when a dataset precomputes activations.\n    subject_lora_path: str | None = None\n\n    @property\n    def effective_subject_model_name(self) -> str:\n',
        1,
    ),
    (
        '        if self.dataset_config.subject_model_name is not None:\n            model_str += "_reads_" + self.dataset_config.subject_model_name.split("/")[-1]\n',
        '        if self.dataset_config.subject_model_name is not None:\n            model_str += "_reads_" + self.dataset_config.subject_model_name.split("/")[-1]\n        if self.dataset_config.subject_lora_path is not None:\n            model_str += "_lora_" + self.dataset_config.subject_lora_path.rstrip("/").split("/")[-1]\n',
        1,
    ),
])

# --------------------------------------------------------------------------------------
# 7g. subject_lora_path in dataset building (classification)
# --------------------------------------------------------------------------------------
patch('nl_probes/dataset_classes/classification.py', [
    (
        '                model_kwargs=self.model_kwargs,\n                model=self.model,\n                subject_tokenizer=subject_tokenizer,\n            )\n',
        '                model_kwargs=self.model_kwargs,\n                model=self.model,\n                lora_path=self.dataset_config.subject_lora_path,  # precomputed acts come from base+adapter\n                subject_tokenizer=subject_tokenizer,\n            )\n',
        1,
    ),
    (
        '    if lora_path is not None:\n        model = PeftModel.from_pretrained(model, lora_path)\n',
        '    if lora_path is not None and model is not None:  # only meaningful when activations are precomputed\n        model = PeftModel.from_pretrained(model, lora_path)\n        model.eval()\n',
        1,
    ),
])

# --------------------------------------------------------------------------------------
# 7h. Pad PRECOMPUTED subject vectors to the oracle width (eval-on-start crash fix)
# --------------------------------------------------------------------------------------
patch('nl_probes/utils/dataset_utils.py', [
    (
        '    # Select datapoints that need generation\n    to_fill: list[tuple[int, TrainingDataPoint]] = [\n',
        '    # [pad_precomputed] Cross-model: vectors that are ALREADY present (e.g. classification test splits built with\n    # save_acts=True) were captured at the SUBJECT\'s width; pad them to the ORACLE\'s width before anything else,\n    # otherwise the early return below lets a 4096-wide vector reach a 5120-wide hook.\n    if subject_model is not None:\n        d_o = int(oracle_hidden_size) if oracle_hidden_size is not None else int(model.config.hidden_size)\n        repadded: list[TrainingDataPoint] = []\n        for dp in batch_points:\n            v = dp.steering_vectors\n            if v is not None and v.shape[1] != d_o:\n                dp = dp.model_copy(deep=True)\n                if v.shape[1] > d_o:\n                    dp.steering_vectors = v[:, :d_o].contiguous()  # [truncate] lossy: keep the first d_o dims\n                else:\n                    dp.steering_vectors = torch.nn.functional.pad(v, (0, d_o - v.shape[1]))\n            repadded.append(dp)\n        batch_points = repadded\n\n    # Select datapoints that need generation\n    to_fill: list[tuple[int, TrainingDataPoint]] = [\n',
        1,
    ),
])

# --------------------------------------------------------------------------------------
# 8. sft.py: load a frozen subject, thread it through preflight / train loop / eval; env-driven main
# --------------------------------------------------------------------------------------
sft = ROOT / "nl_probes" / "sft.py"
text = sft.read_text(encoding="utf-8")

# 8a. every `model_name=model_name,` kwarg gets a companion `subject_model_name=subject_model_name,`
#     (covers mk_cfg's body, all build_loader_groups calls, the build_loader_groups call in main,
#      and the SelfInterpTrainingConfig(...) construction in main).
n_before = len(re.findall(r"\n(\s+)model_name=model_name,", text))
assert n_before == 11, f"[sft.py] expected 11 `model_name=model_name,` kwargs, found {n_before}"
text = re.sub(
    r"\n(\s+)model_name=model_name,",
    lambda m: f"\n{m.group(1)}model_name=model_name,\n{m.group(1)}subject_model_name=subject_model_name,",
    text,
)

edits_sft: list[tuple[str, str, int]] = [
    # eval_all_datasets: signature + pass-through
    (
        "    dtype: torch.dtype,\n"
        "    global_step: int,\n"
        ") -> None:\n"
        "    model.eval()\n"
        "    eval_results = {}\n",
        "    dtype: torch.dtype,\n"
        "    global_step: int,\n"
        "    subject_model: torch.nn.Module | None = None,\n"
        "    subject_tokenizer: PreTrainedTokenizer | None = None,\n"
        ") -> None:\n"
        "    model.eval()\n"
        "    eval_results = {}\n",
        1,
    ),
    (
        "            steering_coefficient=cfg.steering_coefficient,\n"
        "            generation_kwargs=cfg.generation_kwargs,\n"
        "        )\n"
        "        percent_format_correct, percent_ans_correct = score_eval_responses(eval_responses, eval_datasets[ds])\n",
        "            steering_coefficient=cfg.steering_coefficient,\n"
        "            generation_kwargs=cfg.generation_kwargs,\n"
        "            subject_model=subject_model,\n"
        "            subject_tokenizer=subject_tokenizer,\n"
        "        )\n"
        "        percent_format_correct, percent_ans_correct = score_eval_responses(eval_responses, eval_datasets[ds])\n",
        1,
    ),
    # oom_preflight_check: signature + pass-through
    (
        "    tokenizer: PreTrainedTokenizer,\n"
        "    device: torch.device,\n"
        "    dtype: torch.dtype,\n"
        ") -> None:\n"
        "    longest_prompt = max(training_data, key=lambda x: len(x.input_ids))\n"
        "    long_prompts = [longest_prompt] * cfg.train_batch_size\n"
        "    long_prompts = materialize_missing_steering_vectors(long_prompts, tokenizer, model)\n",
        "    tokenizer: PreTrainedTokenizer,\n"
        "    device: torch.device,\n"
        "    dtype: torch.dtype,\n"
        "    subject_model: torch.nn.Module | None = None,\n"
        "    subject_tokenizer: PreTrainedTokenizer | None = None,\n"
        ") -> None:\n"
        "    longest_prompt = max(training_data, key=lambda x: len(x.input_ids))\n"
        "    long_prompts = [longest_prompt] * cfg.train_batch_size\n"
        "    long_prompts = materialize_missing_steering_vectors(\n"
        "        long_prompts, tokenizer, model, subject_model=subject_model, subject_tokenizer=subject_tokenizer\n"
        "    )\n",
        1,
    ),
    # train_model: load the frozen subject right after the oracle
    (
        "    set_seed(cfg.seed)\n"
        "    model = load_model(cfg.model_name, dtype, **model_kwargs)\n"
        "\n"
        "    model.enable_input_require_grads()\n",
        "    set_seed(cfg.seed)\n"
        "    model = load_model(cfg.model_name, dtype, **model_kwargs)\n"
        "\n"
        "    # Cross-model: a separate, frozen SUBJECT whose activations the oracle reads (same GPU as the oracle).\n"
        "    subject_model: torch.nn.Module | None = None\n"
        "    subject_tokenizer: PreTrainedTokenizer | None = None\n"
        "    if cfg.subject_model_name is not None:\n"
        "        print(f\"Cross-model: oracle={cfg.model_name}  subject={cfg.subject_model_name}  act_layers={cfg.act_layers}\")\n"
        "        subject_model = load_model(cfg.subject_model_name, dtype, **model_kwargs)\n"
        "        if cfg.subject_lora_path is not None:\n"
        "            subject_model = PeftModel.from_pretrained(subject_model, cfg.subject_lora_path, is_trainable=False)\n"
        "            print(f\"Loaded subject LoRA (kept active during capture): {cfg.subject_lora_path}\")\n"
        "        subject_model.eval()\n"
        "        for p in subject_model.parameters():\n"
        "            p.requires_grad_(False)\n"
        "        subject_tokenizer = load_tokenizer(cfg.subject_model_name)\n"
        "        d_oracle = model.config.hidden_size\n"
        "        d_subject = subject_model.get_base_model().config.hidden_size if isinstance(subject_model, PeftModel) else subject_model.config.hidden_size\n"
        "        assert d_oracle == d_subject, (\n"
        "            f\"Adapter-free cross-model requires matching hidden_size; oracle={d_oracle}, subject={d_subject}\"\n"
        "        )\n"
        "\n"
        "    model.enable_input_require_grads()\n",
        1,
    ),
    (
        "    oom_preflight_check(cfg, training_data, model, submodule, tokenizer, device, dtype)\n",
        "    oom_preflight_check(\n"
        "        cfg, training_data, model, submodule, tokenizer, device, dtype,\n"
        "        subject_model=subject_model, subject_tokenizer=subject_tokenizer,\n"
        "    )\n",
        1,
    ),
    (
        "            t_batch_list = materialize_missing_steering_vectors(t_batch_list, tokenizer, model)\n",
        "            t_batch_list = materialize_missing_steering_vectors(\n"
        "                t_batch_list, tokenizer, model, subject_model=subject_model, subject_tokenizer=subject_tokenizer\n"
        "            )\n",
        1,
    ),
    (
        "eval_all_datasets(cfg, eval_datasets, model, tokenizer, submodule, device, dtype, global_step)\n",
        "eval_all_datasets(\n"
        "                            cfg, eval_datasets, model, tokenizer, submodule, device, dtype, global_step,\n"
        "                            subject_model=subject_model, subject_tokenizer=subject_tokenizer,\n"
        "                        )\n",
        2,
    ),
    # mk_cfg + build_loader_groups signatures
    (
        "    splits: list[str],\n"
        "    model_name: str,\n"
        "    layer_percents: list[int],\n"
        "    save_acts: bool,\n"
        "    batch_size: int,\n"
        ") -> DatasetLoaderConfig:\n",
        "    splits: list[str],\n"
        "    model_name: str,\n"
        "    layer_percents: list[int],\n"
        "    save_acts: bool,\n"
        "    batch_size: int,\n"
        "    subject_model_name: str | None = None,\n"
        ") -> DatasetLoaderConfig:\n",
        1,
    ),
    (
        "def build_loader_groups(\n"
        "    *,\n"
        "    model_name: str,\n",
        "def build_loader_groups(\n"
        "    *,\n"
        "    model_name: str,\n"
        "    subject_model_name: str | None = None,\n",
        1,
    ),
    # env-driven debug mode for smoke tests
    (
        "    DEBUG = False\n"
        "    num_datapoints = 100_000\n"
        "\n"
        "    # DEBUG = True\n",
        "    DEBUG = os.environ.get(\"AO_DEBUG\", \"0\") == \"1\"\n"
        "    num_datapoints = 100_000\n",
        1,
    ),
    # main: model selection via env; subject; attention override; debug sizes
    (
        "    main_train_size = 6000\n"
        "    main_test_size = 250\n",
        "    main_train_size = 6000\n"
        "    main_test_size = 250\n"
        "    if os.environ.get(\"AO_DEBUG\", \"0\") == \"1\":\n"
        "        print(\"AO_DEBUG=1: tiny classification splits for a smoke test\")\n"
        "        main_train_size = 64\n"
        "        main_test_size = 16\n",
        1,
    ),
    (
        "    models = [\n"
        "        # \"Qwen/Qwen3-14B\",\n"
        "        # \"google/gemma-2-27b-it\",\n"
        "        # \"meta-llama/Llama-3.1-8B-Instruct\",\n"
        "        # \"google/gemma-3-4b-it\",\n"
        "        # \"google/gemma-3-12b-it\",\n"
        "        # \"google/gemma-3-27b-it\",\n"
        "        \"Qwen/Qwen3-4B\",\n"
        "    ]\n"
        "\n"
        "    for model_name in models:\n"
        "        hf_repo_name = \"N/A\"\n"
        "\n"
        "        model_name_str = model_name.split(\"/\")[-1].replace(\".\", \"_\").replace(\" \", \"_\")\n",
        "    # ---------------- Cross-model configuration (env-driven so the pod launch is a one-liner) ----------------\n"
        "    #   AO_ORACLE_MODEL  : backbone that carries the LoRA          (default meta-llama/Llama-3.1-8B-Instruct)\n"
        "    #   AO_SUBJECT_MODEL : model whose activations are read        (default Qwen/Qwen3-8B; \"\" -> self-oracle)\n"
        "    #   AO_SUBJECT_LORA  : optional LoRA on the subject (Taboo)     (default none)\n"
        "    #   AO_ATTN          : attention impl override, e.g. sdpa        (default: upstream picks flash_attention_2)\n"
        "    #   AO_DEBUG=1       : tiny datasets for a smoke test\n"
        "    subject_model_name: str | None = os.environ.get(\"AO_SUBJECT_MODEL\", \"Qwen/Qwen3-8B\") or None\n"
        "    subject_lora_path: str | None = os.environ.get(\"AO_SUBJECT_LORA\", \"\") or None\n"
        "\n"
        "    models = [\n"
        "        os.environ.get(\"AO_ORACLE_MODEL\", \"meta-llama/Llama-3.1-8B-Instruct\"),\n"
        "    ]\n"
        "\n"
        "    for model_name in models:\n"
        "        hf_repo_name = \"N/A\"\n"
        "\n"
        "        model_name_str = model_name.split(\"/\")[-1].replace(\".\", \"_\").replace(\" \", \"_\")\n"
        "        if subject_model_name is not None:\n"
        "            model_name_str += \"_reads_\" + subject_model_name.split(\"/\")[-1].replace(\".\", \"_\").replace(\" \", \"_\")\n"
        "        if subject_lora_path:  # organism identity in the save dir: clock vs leaf runs must not collide\n"
        "            model_name_str += \"_lora_\" + subject_lora_path.rstrip(\"/\").split(\"/\")[-1].replace(\".\", \"_\").replace(\" \", \"_\")\n",
        1,
    ),
    (
        "        train_batch_size = 16\n"
        "        gradient_checkpointing = True\n"
        "        model_kwargs = {}\n",
        "        train_batch_size = 16\n"
        "        gradient_checkpointing = True\n"
        "        model_kwargs = {}\n"
        "        if os.environ.get(\"AO_ATTN\"):\n"
        "            model_kwargs[\"attn_implementation\"] = os.environ[\"AO_ATTN\"]\n",
        1,
    ),
    (
        "            cfg = SelfInterpTrainingConfig(\n"
        "                model_name=model_name,\n"
        "                subject_model_name=subject_model_name,\n"
        "                hook_onto_layer=hook_layer,\n",
        "            cfg = SelfInterpTrainingConfig(\n"
        "                model_name=model_name,\n"
        "                subject_model_name=subject_model_name,\n"
        "                subject_lora_path=subject_lora_path,\n"
        "                hook_onto_layer=hook_layer,\n",
        1,
    ),
    # smoke-test switch: drop past-lens (the only gated source) from the mixture
    (
        '                "dataset_loaders": latentqa_loaders + classification_dataset_loaders + past_lens_loaders,\n',
        '                # AO_SMOKE_NO_PASTLENS=1 drops the only gated data source (lmsys) so a smoke test needs no HF token\n                "dataset_loaders": latentqa_loaders\n                + classification_dataset_loaders\n                + ([] if os.environ.get("AO_SMOKE_NO_PASTLENS", "0") == "1" else past_lens_loaders),\n',
        1,
    ),
    # AO_DEBUG: cap training data so a smoke test finishes in minutes (LatentQA ignores num_train)
    (
        '            all_training_data, all_eval_data = build_datasets(\n                cfg, dataset_loaders=loop_dataset_loaders, window_mult=cfg.window_mult\n            )\n',
        '            all_training_data, all_eval_data = build_datasets(\n                cfg, dataset_loaders=loop_dataset_loaders, window_mult=cfg.window_mult\n            )\n            if os.environ.get("AO_DEBUG", "0") == "1":\n                all_training_data = all_training_data[: 20 * cfg.train_batch_size]\n                print(f"AO_DEBUG=1: truncated training data to {len(all_training_data)} examples (20 batches)")\n',
        1,
    ),
    # injection recipe plumbing (normalize flag, hook layer by percent, env vars)
    (
        '        steering_coefficient=cfg.steering_coefficient,\n        device=device,\n        dtype=dtype,\n    )\n\n    tokenized_input = {\n        "input_ids": training_batch.input_ids,\n',
        '        steering_coefficient=cfg.steering_coefficient,\n        device=device,\n        dtype=dtype,\n        normalize=cfg.injection_mode == "norm_matched",\n    )\n\n    tokenized_input = {\n        "input_ids": training_batch.input_ids,\n',
        1,
    ),
    (
        '            subject_model=subject_model,\n            subject_tokenizer=subject_tokenizer,\n        )\n        percent_format_correct, percent_ans_correct = score_eval_responses(eval_responses, eval_datasets[ds])\n',
        '            subject_model=subject_model,\n            subject_tokenizer=subject_tokenizer,\n            normalize_injection=cfg.injection_mode == "norm_matched",\n        )\n        percent_format_correct, percent_ans_correct = score_eval_responses(eval_responses, eval_datasets[ds])\n',
        1,
    ),
    (
        '    submodule = get_hf_submodule(model, cfg.hook_onto_layer)\n\n    if cfg.use_lora and cfg.load_lora_path is None:\n',
        '    submodule = get_hf_submodule(model, cfg.hook_onto_layer)\n    print(f"Injection: mode={cfg.injection_mode} lambda={cfg.steering_coefficient} at ORACLE layer {cfg.hook_onto_layer}")\n\n    if cfg.use_lora and cfg.load_lora_path is None:\n',
        1,
    ),
    (
        '    subject_lora_path: str | None = os.environ.get("AO_SUBJECT_LORA", "") or None\n',
        '    subject_lora_path: str | None = os.environ.get("AO_SUBJECT_LORA", "") or None\n    #   AO_INJECTION         : norm_matched (Karvonen default) | raw (Bersia & Gaintseva App. B.3)\n    #   AO_HOOK_LAYER_PERCENT: e.g. 50 -> inject at 50% of the ORACLE depth (Qwen3-8B L18, Llama L16); unset -> layer 1\n    #   AO_LAMBDA            : steering coefficient (default 1.0)\n    injection_mode: str = os.environ.get("AO_INJECTION", "norm_matched")\n    hook_layer_percent: int | None = (\n        int(os.environ["AO_HOOK_LAYER_PERCENT"]) if os.environ.get("AO_HOOK_LAYER_PERCENT") else None\n    )\n    steering_lambda: float = float(os.environ.get("AO_LAMBDA", "1.0"))\n',
        1,
    ),
    (
        '                subject_lora_path=subject_lora_path,\n                hook_onto_layer=hook_layer,\n',
        '                subject_lora_path=subject_lora_path,\n                hook_onto_layer=hook_layer,\n                hook_onto_layer_percent=hook_layer_percent,\n                injection_mode=injection_mode,\n                steering_coefficient=steering_lambda,\n',
        1,
    ),
    # zero-padding: replace the equal-hidden-size assert
    (
        '        assert d_oracle == d_subject, (\n            f"Adapter-free cross-model requires matching hidden_size; oracle={d_oracle}, subject={d_subject}"\n        )\n',
        '        if d_subject < d_oracle:\n            print(f"Cross-model: zero-padding subject activations {d_subject} -> {d_oracle} (lossless)")\n        elif d_subject > d_oracle:\n            print(f"Cross-model: TRUNCATING subject activations {d_subject} -> {d_oracle} (lossy: keeps the first {d_oracle} dims)")\n',
        1,
    ),
    # absolute hook layer via env (AO_HOOK_LAYER)
    (
        '    steering_lambda: float = float(os.environ.get("AO_LAMBDA", "1.0"))\n',
        '    steering_lambda: float = float(os.environ.get("AO_LAMBDA", "1.0"))\n    #   AO_HOOK_LAYER        : ABSOLUTE oracle injection layer (paper App. B.3 says 18 for a Qwen3-8B oracle); default 1\n    #                          (read ABOVE, at `hook_layer = ...`, so it is defined before its first use)\n',
        1,
    ),
    (
        '    hook_layer = 1\n',
        '    hook_layer = int(os.environ.get("AO_HOOK_LAYER", "1"))  # ABSOLUTE oracle injection layer; default 1 (Karvonen). Overridden by hook_onto_layer_percent when that is set.\n',
        1,
    ),
    # AO_REVISION: pin the HF commit for oracle + subject
    (
        '        if os.environ.get("AO_ATTN"):\n            model_kwargs["attn_implementation"] = os.environ["AO_ATTN"]\n',
        '        if os.environ.get("AO_ATTN"):\n            model_kwargs["attn_implementation"] = os.environ["AO_ATTN"]\n        if os.environ.get("AO_REVISION"):\n            # Pin the HF commit for BOTH oracle and subject (released Qwen3-8B oracle base:\n            # b968826d9c46dd6066d109eabc6255188de91218). Leave unset when either model is a local path.\n            model_kwargs["revision"] = os.environ["AO_REVISION"]\n',
        1,
    ),
    # per-model HF revision (AO_REVISION -> subject only; AO_ORACLE_REVISION -> oracle)
    (
        '        if os.environ.get("AO_REVISION"):\n            # Pin the HF commit for BOTH oracle and subject (released Qwen3-8B oracle base:\n            # b968826d9c46dd6066d109eabc6255188de91218). Leave unset when either model is a local path.\n            model_kwargs["revision"] = os.environ["AO_REVISION"]\n',
        '        # Revisions are PER-MODEL. AO_REVISION pins the SUBJECT (e.g. the released oracle\'s Qwen3-8B base commit\n        # b968826d9c46dd6066d109eabc6255188de91218); AO_ORACLE_REVISION pins the oracle. A commit hash from one\n        # repo does not exist in another (Qwen3-8B hash -> Qwen3-14B load fails with \'Unrecognized model\').\n        subject_model_kwargs = dict(model_kwargs)\n        if os.environ.get("AO_REVISION"):\n            subject_model_kwargs["revision"] = os.environ["AO_REVISION"]\n        if os.environ.get("AO_ORACLE_REVISION"):\n            model_kwargs["revision"] = os.environ["AO_ORACLE_REVISION"]\n',
        1,
    ),
    (
        '            classification_datasets=classification_datasets,\n            model_kwargs=model_kwargs,\n',
        '            classification_datasets=classification_datasets,\n            model_kwargs=subject_model_kwargs,  # dataset precompute loads the SUBJECT\n',
        1,
    ),
    (
        '                model_kwargs=model_kwargs,\n                verbose=True,\n            )\n',
        '                model_kwargs=model_kwargs,\n                subject_model_kwargs=subject_model_kwargs,\n                verbose=True,\n            )\n',
        1,
    ),
    (
        '    model_kwargs: dict[str, Any],\n    verbose: bool = False,\n):\n',
        '    model_kwargs: dict[str, Any],\n    verbose: bool = False,\n    subject_model_kwargs: dict[str, Any] | None = None,\n):\n',
        1,
    ),
    (
        '        subject_model = load_model(cfg.subject_model_name, dtype, **model_kwargs)\n',
        '        skw = {**(subject_model_kwargs if subject_model_kwargs is not None else model_kwargs),\n               "device_map": {"": f"cuda:{local_rank}"}}\n        subject_model = load_model(cfg.subject_model_name, dtype, **skw)\n',
        1,
    ),
]
for old, new, n in edits_sft:
    c = text.count(old)
    assert c == n, f"[sft.py] expected {n} occurrence(s), found {c}:\n{old!r}"
    text = text.replace(old, new)
# 7g. subject_lora_path: signatures + companion kwargs
_sig = "    subject_model_name: str | None = None,\n"
assert text.count(_sig) == 2, text.count(_sig)
text = text.replace(_sig, _sig + "    subject_lora_path: str | None = None,\n")
_lines = text.split("\n"); _out = []; _ins = 0
for _i, _l in enumerate(_lines):
    _out.append(_l)
    _m = re.match(r"^(\s+)subject_model_name=subject_model_name,$", _l)
    if _m and "subject_lora_path" not in (_lines[_i + 1] if _i + 1 < len(_lines) else ""):
        _out.append(f"{_m.group(1)}subject_lora_path=subject_lora_path,"); _ins += 1
text = "\n".join(_out)
assert _ins == 10, _ins
sft.write_text(text, encoding="utf-8")
print(f"patched nl_probes/sft.py  ({len(edits_sft) + 1} edit(s))")

# --------------------------------------------------------------------------------------
# 9. Resume from a saved adapter: AO_RESUME_ADAPTER + AO_RESUME_STEP (fast-forward data order and LR schedule)
# --------------------------------------------------------------------------------------
patch('nl_probes/sft.py', [
    (
        '                "load_lora_path": None,\n                # AO_SMOKE_NO_PASTLENS=1 drops the only gated data source (lmsys) so a smoke test needs no HF token\n',
        '                "load_lora_path": os.environ.get("AO_RESUME_ADAPTER") or None,  # [resume] load this adapter as trainable\n                # AO_SMOKE_NO_PASTLENS=1 drops the only gated data source (lmsys) so a smoke test needs no HF token\n',
        1,
    ),
    (
        '    global_step = 0\n',
        '    global_step = 0\n    RESUME_STEP = int(os.environ.get("AO_RESUME_STEP", "0"))  # [resume] optimizer steps already done by the loaded adapter\n    if RESUME_STEP and rank == 0:\n        print(f"[resume] adapter={cfg.load_lora_path} fast-forwarding {RESUME_STEP} steps (data order + LR schedule; AdamW state resets)")\n',
        1,
    ),
    (
        '            t_batch_list: list[TrainingDataPoint] = training_data[start : start + cfg.train_batch_size]\n',
        '            # [resume] fast-forward through already-trained steps: no compute; advance the LR schedule and step counter only\n            if RESUME_STEP and global_step < RESUME_STEP:\n                if (step_idx + 1) % cfg.gradient_accumulation_steps == 0:\n                    scheduler.step()\n                    global_step += 1\n                    if global_step == RESUME_STEP and rank == 0:\n                        print(f"[resume] reached step {global_step}; lr={scheduler.get_last_lr()[0]:.3e}; training continues")\n                continue\n            t_batch_list: list[TrainingDataPoint] = training_data[start : start + cfg.train_batch_size]\n',
        1,
    ),
])

print("\nAll patches applied. Review with:  git -C", ROOT, "diff --stat")

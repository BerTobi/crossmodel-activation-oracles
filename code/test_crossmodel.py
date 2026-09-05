"""
CPU-only correctness tests for the cross-model patch. Run from the repo root:

    cd activation_oracles && python ../test_crossmodel.py

What is checked
  T1  LatentQA datapoints: oracle prompt in Llama token space, context in Qwen token space,
      placeholder count == context position count, layers from the SUBJECT's depth.
  T2  Classification datapoints (hand-built, no HF download): same invariants.
  T3  Regression: with subject_model_name=None the functions reduce to upstream behaviour
      (both sides tokenized identically) and the cache filename has no '_reads_' marker.
  T4  materialize_missing_steering_vectors, cross-model path, end to end on two SEPARATE tiny
      model instances (Qwen3-0.6B twice, oracle wrapped in a fresh LoRA). Asserts:
        - shapes / finiteness,
        - vectors equal a direct forward pass through the subject (ground truth),
        - vectors equal the ORIGINAL disable_adapter() path — since the oracle's base weights
          are the same checkpoint as the subject, both paths must read the same thing.
      Set AO_TEST_SKIP_MODELS=1 to skip (downloads ~1.2 GB once).

Oracle tokenizer uses an ungated Llama-3.1 mirror so no HF token is needed.
"""

import os
import sys
from pathlib import Path

REPO = Path.cwd()
assert (REPO / "nl_probes").exists(), "run from the activation_oracles repo root"
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig  # noqa: E402
from nl_probes.dataset_classes.classification import (  # noqa: E402
    ClassificationDatapoint,
    ClassificationDatasetConfig,
    create_vector_dataset,
)
from nl_probes.dataset_classes import latentqa_dataset as lqd  # noqa: E402
from nl_probes.utils.common import layer_percent_to_layer, load_tokenizer  # noqa: E402
from nl_probes.utils.dataset_utils import (  # noqa: E402
    SPECIAL_TOKEN,
    TrainingDataPoint,
    create_training_datapoint,
    materialize_missing_steering_vectors,
)

ORACLE = "NousResearch/Meta-Llama-3.1-8B-Instruct"  # ungated mirror of meta-llama/Llama-3.1-8B-Instruct
SUBJECT = "Qwen/Qwen3-8B"

PASS = "  ✓"
FAIL = "  ✗"
failures = 0


def check(cond: bool, msg: str) -> None:
    global failures
    print((PASS if cond else FAIL), msg)
    if not cond:
        failures += 1


def common_invariants(dp: TrainingDataPoint, otok, stok, sid_oracle: int, subject_layers: set[int], tag: str):
    # placeholders live in the ORACLE prompt and count matches the SUBJECT context positions
    check(len(dp.positions) == len(dp.context_positions), f"[{tag}] |positions| == |context_positions| ({len(dp.positions)})")
    check(all(dp.input_ids[p] == sid_oracle for p in dp.positions), f"[{tag}] every oracle placeholder is Llama ' ?' id {sid_oracle}")
    check(dp.layer in subject_layers, f"[{tag}] layer {dp.layer} is a SUBJECT depth {sorted(subject_layers)}")
    # the oracle prompt decodes cleanly with the oracle tokenizer
    o_txt = otok.decode(dp.input_ids, skip_special_tokens=False)
    check(f"Layer: {dp.layer}" in o_txt, f"[{tag}] oracle prompt carries 'Layer: {dp.layer}'")
    check("<|start_header_id|>" in o_txt, f"[{tag}] oracle prompt is in Llama chat format")
    # the context decodes cleanly with the SUBJECT tokenizer and is NOT Llama-formatted
    s_txt = stok.decode(dp.context_input_ids, skip_special_tokens=False)
    check("<|start_header_id|>" not in s_txt, f"[{tag}] context is NOT Llama-formatted")
    check(max(dp.context_positions) < len(dp.context_input_ids), f"[{tag}] context positions in range")
    return o_txt, s_txt


print("\n=== loading tokenizers ===")
otok = load_tokenizer(ORACLE)
stok = load_tokenizer(SUBJECT)
sid_oracle = otok.encode(SPECIAL_TOKEN, add_special_tokens=False)
sid_subject = stok.encode(SPECIAL_TOKEN, add_special_tokens=False)
check(len(sid_oracle) == 1, f"oracle ' ?' is a single token: {sid_oracle}")
check(len(sid_subject) == 1, f"subject ' ?' is a single token: {sid_subject}")
sid_oracle = sid_oracle[0]
subject_layers = {layer_percent_to_layer(SUBJECT, p) for p in (25, 50, 75)}
oracle_layers = {layer_percent_to_layer(ORACLE, p) for p in (25, 50, 75)}
check(subject_layers == {9, 18, 27}, f"Qwen3-8B 25/50/75% -> {sorted(subject_layers)} (expect 9/18/27)")
check(oracle_layers == {8, 16, 24}, f"Llama-3.1-8B 25/50/75% -> {sorted(oracle_layers)} (expect 8/16/24)")

# ------------------------------------------------------------------------------------------
print("\n=== T1: LatentQA datapoints (oracle=Llama, subject=Qwen) ===")
paths = lqd.latentqa_loader.DataPaths(
    system=None,
    stimulus_completion="datasets/latentqa_datasets/train/stimulus_completion.json",
    stimulus="datasets/latentqa_datasets/train/stimulus.json",
    control="datasets/latentqa_datasets/train/control.json",
    qa="datasets/latentqa_datasets/train/qa.json",
)
ds = lqd.latentqa_loader.load_latentqa_dataset(paths, filter_prefixes=[], train_percent=1.0, add_thought_tokens=False, seed=0)
params = lqd.LatentQADatasetConfig()
layers = sorted(subject_layers)
n_ok = 0
for i, raw in enumerate(ds[i] for i in range(24)):
    dp = lqd.create_latentqa_training_datapoint(raw, otok, layers, params, subject_tokenizer=stok)
    if i < 2:
        o_txt, s_txt = common_invariants(dp, otok, stok, sid_oracle, subject_layers, f"lqa#{i}")
        print(f"    oracle prompt head : {o_txt[:110]!r}")
        print(f"    subject context hd : {s_txt[:110]!r}")
    else:
        ok = (
            len(dp.positions) == len(dp.context_positions)
            and all(dp.input_ids[p] == sid_oracle for p in dp.positions)
            and dp.layer in subject_layers
        )
        n_ok += ok
check(n_ok == 22, f"[lqa] remaining {n_ok}/22 datapoints satisfy all invariants")

# ------------------------------------------------------------------------------------------
print("\n=== T2: Classification datapoints (hand-built, no download) ===")
cls_points = [
    ClassificationDatapoint(
        activation_prompt="The movie was an absolute delight from start to finish.",
        classification_prompt="Is the sentiment of this text positive? Answer yes or no.",
        target_response="yes",
        ds_label="positive",
    ),
    ClassificationDatapoint(
        activation_prompt="I regret every minute I spent watching this film.",
        classification_prompt="Is the sentiment of this text positive? Answer yes or no.",
        target_response="no",
        ds_label="negative",
    ),
]
cls_data = create_vector_dataset(
    cls_points, otok, SUBJECT, batch_size=2, act_layers=layers,
    min_end_offset=-1, max_end_offset=-3, max_window_size=4, min_window_size=1,
    save_acts=False, datapoint_type="classification_sst2", subject_tokenizer=stok,
)
check(len(cls_data) == len(cls_points) * len(layers), f"[cls] {len(cls_data)} datapoints = {len(cls_points)} x {len(layers)} layers")
for i, dp in enumerate(cls_data[:2]):
    o_txt, s_txt = common_invariants(dp, otok, stok, sid_oracle, subject_layers, f"cls#{i}")
    check("<|im_start|>" in s_txt, f"[cls#{i}] context IS Qwen chat-formatted")
    print(f"    subject context: {s_txt[:120]!r}")

# ------------------------------------------------------------------------------------------
print("\n=== T3: Regression — subject_model_name=None reduces to upstream ===")
cfg_self = DatasetLoaderConfig(ClassificationDatasetConfig(classification_dataset_name="sst2"), 8, 0, ["train"], SUBJECT, [50], False, 8)
check(cfg_self.effective_subject_model_name == SUBJECT, "[reg] effective_subject_model_name falls back to model_name")
cfg_cross = DatasetLoaderConfig(ClassificationDatasetConfig(classification_dataset_name="sst2"), 8, 0, ["train"], ORACLE, [50], False, 8, subject_model_name=SUBJECT)
check(cfg_cross.effective_subject_model_name == SUBJECT, "[reg] effective_subject_model_name uses subject when set")
from nl_probes.dataset_classes.classification import ClassificationDatasetLoader  # noqa: E402
fn_self = ClassificationDatasetLoader(cfg_self).get_dataset_filename("train")
fn_cross = ClassificationDatasetLoader(cfg_cross).get_dataset_filename("train")
check("_reads_" not in fn_self, f"[reg] self filename has no _reads_ marker: {fn_self}")
check("_reads_Qwen3-8B" in fn_cross, f"[reg] cross filename carries _reads_Qwen3-8B: {fn_cross}")
# same-tokenizer path: passing subject_tokenizer=None must equal passing the same tokenizer explicitly
import random  # noqa: E402
# the function draws a random window/layer; seed identically so both calls see the same draws
random.seed(123); dp_a = lqd.create_latentqa_training_datapoint(ds[0], stok, layers, params)
random.seed(123); dp_b = lqd.create_latentqa_training_datapoint(ds[0], stok, layers, params, subject_tokenizer=stok)
check(dp_a.model_dump() == dp_b.model_dump(), "[reg] subject_tokenizer=None == same tokenizer explicit (full datapoint equality)")
print("    NOTE: adding the config field changes the cache-filename hash vs upstream (behaviour identical; caches not reused).")

# ------------------------------------------------------------------------------------------
if os.environ.get("AO_TEST_SKIP_MODELS", "0") == "1":
    print("\n=== T4 skipped (AO_TEST_SKIP_MODELS=1) ===")
else:
    print("\n=== T4: materialize_missing_steering_vectors, cross-model path, tiny models on CPU ===")
    from peft import LoraConfig, get_peft_model  # noqa: E402
    from transformers import AutoModelForCausalLM  # noqa: E402

    TINY = "Qwen/Qwen3-0.6B"
    ttok = load_tokenizer(TINY)
    torch.manual_seed(0)
    oracle_base = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32)
    subject = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32)  # SEPARATE instance
    subject.eval()
    for p in subject.parameters():
        p.requires_grad_(False)
    oracle = get_peft_model(
        oracle_base, LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", task_type="CAUSAL_LM")
    )
    oracle.eval()
    d = subject.config.hidden_size
    layer = layer_percent_to_layer(TINY, 50)

    # one datapoint: context in the SUBJECT's token space, oracle prompt in the ORACLE's (same tokenizer here)
    ctx_ids = ttok("The quick brown fox jumps over the lazy dog near the river bank.", add_special_tokens=False)["input_ids"]
    ctx_pos = list(range(len(ctx_ids) - 4, len(ctx_ids)))
    dp = create_training_datapoint(
        datapoint_type="test", prompt="What animal is mentioned?", target_response="fox",
        layer=layer, num_positions=len(ctx_pos), tokenizer=ttok, acts_BD=None, feature_idx=-1,
        context_input_ids=ctx_ids, context_positions=ctx_pos,
    )

    with torch.no_grad():
        # (i) cross-model path
        out_x = materialize_missing_steering_vectors([dp], ttok, oracle, subject_model=subject, subject_tokenizer=ttok)[0]
        # (ii) original path (oracle base weights via disable_adapter)
        out_o = materialize_missing_steering_vectors([dp], ttok, oracle)[0]
        # (iii) ground truth: direct forward through the subject
        hs = subject(torch.tensor([ctx_ids]), output_hidden_states=True).hidden_states
        gt = hs[layer + 1][0, ctx_pos, :]

    vx, vo = out_x.steering_vectors, out_o.steering_vectors
    check(tuple(vx.shape) == (len(ctx_pos), d), f"[mat] cross-model shape {tuple(vx.shape)} == ({len(ctx_pos)}, {d})")
    check(torch.isfinite(vx).all().item(), "[mat] cross-model vectors finite")
    check(torch.allclose(vx.float(), gt.float(), atol=1e-4, rtol=1e-4), f"[mat] cross-model == direct subject forward (max|d|={(vx.float()-gt.float()).abs().max():.2e})")
    check(torch.allclose(vx.float(), vo.float(), atol=1e-4, rtol=1e-4), f"[mat] cross-model == original disable_adapter path (max|d|={(vx.float()-vo.float()).abs().max():.2e})")
    check(out_o.steering_vectors is not None and dp.steering_vectors is None, "[mat] input datapoint untouched; output carries vectors")

print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} FAILURE(S)'}")
sys.exit(1 if failures else 0)

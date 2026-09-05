"""
Does cross-model capture read a PEFT-wrapped subject WITH its adapter active?

Builds a tiny subject = Qwen3-0.6B + random LoRA (B matrices perturbed so the adapter actually changes
the forward), saves it, reloads via PeftModel.from_pretrained (the real code path), then checks that
materialize_missing_steering_vectors(subject_model=<peft subject>) returns vectors equal to a forward
through the PEFT subject and DIFFERENT from the bare base. Also exercises the dataset-side
subject_lora_path plumbing (config field, filename marker).

Run from the repo root:  cd activation_oracles && python ../test_subject_lora.py
"""
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path.cwd()
assert (REPO / "nl_probes").exists(), "run from the activation_oracles repo root"
sys.path.insert(0, str(REPO))

from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig  # noqa: E402
from nl_probes.dataset_classes.classification import ClassificationDatasetConfig, ClassificationDatasetLoader  # noqa: E402
from nl_probes.utils.activation_utils import get_hf_submodule  # noqa: E402
from nl_probes.utils.common import layer_percent_to_layer, load_tokenizer  # noqa: E402
from nl_probes.utils.dataset_utils import create_training_datapoint, materialize_missing_steering_vectors  # noqa: E402

ok = True


def chk(cond, msg):
    global ok
    ok &= bool(cond)
    print(("  PASS " if cond else "  FAIL ") + msg)


TINY = "Qwen/Qwen3-0.6B"
tok = load_tokenizer(TINY)
torch.manual_seed(0)
layer = layer_percent_to_layer(TINY, 50)

# --- build a subject adapter whose effect is non-zero, save it, reload the way sft.py does ---
base_for_adapter = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32)
peft_tmp = get_peft_model(base_for_adapter, LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", task_type="CAUSAL_LM"))
with torch.no_grad():
    for n, p in peft_tmp.named_parameters():
        if "lora_B" in n:
            p.add_(torch.randn_like(p) * 0.02)  # default init has B=0 -> identity; make the adapter bite
tmpdir = tempfile.mkdtemp(prefix="subj_lora_")
peft_tmp.save_pretrained(tmpdir)
del peft_tmp, base_for_adapter

subject_base = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32).eval()
subject = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32), tmpdir, is_trainable=False).eval()
oracle = get_peft_model(AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32),
                        LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", task_type="CAUSAL_LM")).eval()

print("=== submodule lookup on a PEFT subject ===")
sm = get_hf_submodule(subject, layer, use_lora=True)
chk(sm is subject.base_model.model.model.layers[layer], f"get_hf_submodule(peft, {layer}, use_lora=True) -> base_model.model.model.layers[{layer}]")

ctx_ids = tok("The adapter must be active while the subject is read.", add_special_tokens=False)["input_ids"]
ctx_pos = list(range(len(ctx_ids) - 4, len(ctx_ids)))
dp = create_training_datapoint(
    datapoint_type="test", prompt="What is this about?", target_response="adapters",
    layer=layer, num_positions=len(ctx_pos), tokenizer=tok, acts_BD=None, feature_idx=-1,
    context_input_ids=ctx_ids, context_positions=ctx_pos,
)

print("=== capture from PEFT subject ===")
with torch.no_grad():
    got = materialize_missing_steering_vectors([dp], tok, oracle, subject_model=subject, subject_tokenizer=tok)[0].steering_vectors
    ids = torch.tensor([ctx_ids])
    gt_peft = subject(ids, output_hidden_states=True).hidden_states[layer + 1][0, ctx_pos, :]
    gt_base = subject_base(ids, output_hidden_states=True).hidden_states[layer + 1][0, ctx_pos, :]

chk(tuple(got.shape) == (len(ctx_pos), subject_base.config.hidden_size), f"shape {tuple(got.shape)}")
chk(torch.allclose(got.float(), gt_peft.float(), atol=1e-4), f"== forward through PEFT subject (max|d|={(got.float()-gt_peft.float()).abs().max():.2e})")
chk(not torch.allclose(got.float(), gt_base.float(), atol=1e-3), f"!= forward through bare base (max|d|={(got.float()-gt_base.float()).abs().max():.2e}) -> adapter was ACTIVE")
chk(subject.training is False, "subject left in eval mode")

print("=== dataset-side plumbing ===")
cfg = DatasetLoaderConfig(ClassificationDatasetConfig(classification_dataset_name="sst2"), 8, 0, ["train"],
                          "Qwen/Qwen3-14B", [50], False, 8, subject_model_name="Qwen/Qwen3-8B", subject_lora_path="/x/y/Mpp_clock_adapter")
fn = ClassificationDatasetLoader(cfg).get_dataset_filename("train")
chk("_reads_Qwen3-8B" in fn and "_lora_Mpp_clock_adapter" in fn, f"filename carries subject + adapter markers: {fn}")
cfg0 = DatasetLoaderConfig(ClassificationDatasetConfig(classification_dataset_name="sst2"), 8, 0, ["train"], "Qwen/Qwen3-8B", [50], False, 8)
chk(cfg0.subject_lora_path is None and "_lora_" not in ClassificationDatasetLoader(cfg0).get_dataset_filename("train"), "default: no adapter marker  [regression]")

print("\nALL PASSED" if ok else "\nFAILURES")
sys.exit(0 if ok else 1)

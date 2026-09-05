"""
Zero-padding test for subject hidden_size < oracle hidden_size. CPU, uses the cached Qwen3-0.6B
(d=1024) as the subject and passes an explicit larger oracle_hidden_size so no second model is needed.

Run from the repo root:  cd activation_oracles && python ../test_padding.py
"""
import sys
from pathlib import Path

import torch

REPO = Path.cwd()
assert (REPO / "nl_probes").exists(), "run from the activation_oracles repo root"
sys.path.insert(0, str(REPO))

from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

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
oracle = get_peft_model(
    AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32),
    LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", task_type="CAUSAL_LM"),
)
subject = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32)
subject.eval()
d_s = subject.config.hidden_size
d_o = d_s + 256  # pretend the oracle is wider (e.g. 4096 -> 5120 in the real run)
layer = layer_percent_to_layer(TINY, 50)

ctx_ids = tok("Zero padding should be lossless for the leading dimensions.", add_special_tokens=False)["input_ids"]
ctx_pos = list(range(len(ctx_ids) - 3, len(ctx_ids)))
dp = create_training_datapoint(
    datapoint_type="test", prompt="What is this about?", target_response="padding",
    layer=layer, num_positions=len(ctx_pos), tokenizer=tok, acts_BD=None, feature_idx=-1,
    context_input_ids=ctx_ids, context_positions=ctx_pos,
)

with torch.no_grad():
    padded = materialize_missing_steering_vectors([dp], tok, oracle, subject_model=subject, subject_tokenizer=tok, oracle_hidden_size=d_o)[0]
    same = materialize_missing_steering_vectors([dp], tok, oracle, subject_model=subject, subject_tokenizer=tok)[0]  # auto: d_o == d_s here
    gt = subject(torch.tensor([ctx_ids]), output_hidden_states=True).hidden_states[layer + 1][0, ctx_pos, :]

v = padded.steering_vectors
print("=== zero-padding (subject narrower than oracle) ===")
chk(tuple(v.shape) == (len(ctx_pos), d_o), f"padded shape {tuple(v.shape)} == ({len(ctx_pos)}, {d_o})")
chk(torch.allclose(v[:, :d_s].float(), gt.float(), atol=1e-4), "leading d_s dims == direct subject forward (lossless)")
chk(torch.equal(v[:, d_s:], torch.zeros(len(ctx_pos), d_o - d_s)), f"trailing {d_o - d_s} dims are exactly zero")
chk(tuple(same.steering_vectors.shape) == (len(ctx_pos), d_s), "auto oracle_hidden_size with equal widths: no padding  [regression]")

print("=== truncation deliberately rejected ===")
with torch.no_grad():
    tr = materialize_missing_steering_vectors([dp], tok, oracle, subject_model=subject, subject_tokenizer=tok, oracle_hidden_size=d_s - 64)
chk(tr[0].steering_vectors.shape[1] == d_s - 64, "subject wider than oracle -> captured vectors truncated to oracle width")

print("\nALL PASSED" if ok else "\nFAILURES")
sys.exit(0 if ok else 1)

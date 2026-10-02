"""
Regression test for the eval-on-start crash: datapoints that ALREADY carry subject-width vectors must be
padded to the oracle width when a subject_model is given, even though nothing needs capturing.

Run from the repo root:  cd activation_oracles && python ../test_padding_precomputed.py
"""
import sys
from pathlib import Path

import torch

REPO = Path.cwd()
assert (REPO / "nl_probes").exists(), "run from the activation_oracles repo root"
sys.path.insert(0, str(REPO))

from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from nl_probes.utils.dataset_utils import TrainingDataPoint, materialize_missing_steering_vectors  # noqa: E402

ok = True


def chk(cond, msg):
    global ok
    ok &= bool(cond)
    print(("  PASS " if cond else "  FAIL ") + msg)


TINY = "Qwen/Qwen3-0.6B"
oracle = get_peft_model(AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32),
                        LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", task_type="CAUSAL_LM")).eval()
subject = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32).eval()
d_s = subject.config.hidden_size
d_o = d_s + 256


def dp_with_vectors(d):
    return TrainingDataPoint(datapoint_type="t", input_ids=[1, 2, 3, 4], labels=[-100, -100, -100, 4], layer=14,
                             steering_vectors=torch.randn(3, d), positions=[0, 1, 2], feature_idx=-1, target_output="x",
                             context_input_ids=None, context_positions=None, ds_label=None)


batch = [dp_with_vectors(d_s), dp_with_vectors(d_s)]
print("=== precomputed vectors, nothing to capture ===")
with torch.no_grad():
    out = materialize_missing_steering_vectors(batch, None, oracle, subject_model=subject, subject_tokenizer=None, oracle_hidden_size=d_o)
chk(all(tuple(d.steering_vectors.shape) == (3, d_o) for d in out), f"all precomputed vectors padded to width {d_o}")
chk(torch.equal(out[0].steering_vectors[:, :d_s], batch[0].steering_vectors), "leading dims preserved")
chk(torch.equal(out[0].steering_vectors[:, d_s:], torch.zeros(3, d_o - d_s)), "trailing dims zero")
chk(batch[0].steering_vectors.shape[1] == d_s, "input datapoints untouched (deep copy)")

print("=== regression: no subject (self-oracle) -> untouched ===")
with torch.no_grad():
    same = materialize_missing_steering_vectors(batch, None, oracle)
chk(all(d is s for d, s in zip(same, batch)), "self-oracle path returns the same objects, no padding")

print("=== regression: equal widths -> untouched ===")
with torch.no_grad():
    eq = materialize_missing_steering_vectors(batch, None, oracle, subject_model=subject, subject_tokenizer=None)
chk(all(d is s for d, s in zip(eq, batch)), "equal widths: same objects, no copy")

print("=== wider precomputed vectors truncated ===")
wide = dp_with_vectors(d_o)
with torch.no_grad():
    tr = materialize_missing_steering_vectors([wide], None, oracle, subject_model=subject, subject_tokenizer=None, oracle_hidden_size=d_s)
chk(tuple(tr[0].steering_vectors.shape) == (3, d_s) and torch.equal(tr[0].steering_vectors, wide.steering_vectors[:, :d_s]), "truncated to oracle width, leading dims preserved")
chk(wide.steering_vectors.shape[1] == d_o, "input untouched (deep copy)")

print("\nALL PASSED" if ok else "\nFAILURES")
sys.exit(0 if ok else 1)

"""
Unit tests for the injection-recipe options. CPU only, no model download (config checks hit HF for
two small config.json files, already cached from earlier tests).

Run from the repo root:  cd activation_oracles && python ../test_injection.py
"""
import sys
from pathlib import Path

import torch

REPO = Path.cwd()
assert (REPO / "nl_probes").exists(), "run from the activation_oracles repo root"
sys.path.insert(0, str(REPO))

from nl_probes.configs.sft_config import SelfInterpTrainingConfig  # noqa: E402
from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook  # noqa: E402

ok = True


def chk(cond, msg):
    global ok
    ok &= bool(cond)
    print(("  PASS " if cond else "  FAIL ") + msg)


torch.manual_seed(0)


class Dummy(torch.nn.Module):
    def forward(self, x):
        return (x.clone(),)  # decoder-layer-style tuple output


B, L, D = 2, 7, 16
x = torch.randn(B, L, D)
v = [torch.randn(3, D) * 5, torch.randn(2, D) * 5]  # deliberately NOT unit norm
pos = [[1, 2, 3], [4, 5]]
m = Dummy()
dev, f32 = torch.device("cpu"), torch.float32


def run(**kw):
    hook = get_hf_activation_steering_hook(v, pos, 1.0, dev, f32, **kw)
    with add_hook(m, hook):
        return m(x)[0]


print("=== hook modes ===")
out_raw, out_nm, out_default = run(normalize=False), run(normalize=True), run()

exp_raw = x.clone()
exp_raw[0, pos[0]] += v[0]
exp_raw[1, pos[1]] += v[1]
exp_nm = x.clone()
for b in range(B):
    o = x[b, pos[b]]
    exp_nm[b, pos[b]] = o + torch.nn.functional.normalize(v[b], dim=-1) * o.norm(dim=-1, keepdim=True)

chk(torch.allclose(out_raw, exp_raw, atol=1e-6), "raw mode: resid[pos] == orig + v  (exact)")
chk(torch.allclose(out_nm, exp_nm, atol=1e-6), "norm_matched mode: resid[pos] == orig + normalize(v)*||orig||")
chk(torch.allclose(out_default, out_nm), "default (no flag) == norm_matched   [upstream regression]")
untouched = [i for i in range(L) if i not in pos[0]]
chk(torch.equal(out_raw[0, untouched], x[0, untouched]) and torch.equal(out_nm[0, untouched], x[0, untouched]),
    "non-placeholder positions untouched in both modes")
chk(not torch.allclose(out_raw, out_nm), "raw and norm_matched differ when ||v|| != ||orig||  (sanity)")

# lambda scaling in raw mode
hook2 = get_hf_activation_steering_hook(v, pos, 0.5, dev, f32, normalize=False)
with add_hook(m, hook2):
    out_half = m(x)[0]
exp_half = x.clone()
exp_half[0, pos[0]] += 0.5 * v[0]
exp_half[1, pos[1]] += 0.5 * v[1]
chk(torch.allclose(out_half, exp_half, atol=1e-6), "raw mode honours lambda (0.5)")

print("=== config: injection layer by ORACLE depth ===")
c1 = SelfInterpTrainingConfig(model_name="Qwen/Qwen3-8B", hook_onto_layer_percent=50, injection_mode="raw").finalize([])
c2 = SelfInterpTrainingConfig(
    model_name="NousResearch/Meta-Llama-3.1-8B-Instruct", subject_model_name="Qwen/Qwen3-8B",
    hook_onto_layer_percent=50, injection_mode="raw",
).finalize([])
c3 = SelfInterpTrainingConfig(model_name="Qwen/Qwen3-8B").finalize([])
chk(c1.hook_onto_layer == 18, f"Qwen3-8B oracle @50% -> hook layer {c1.hook_onto_layer}  (expect 18)")
chk(c2.hook_onto_layer == 16, f"Llama-3.1-8B oracle @50% -> hook layer {c2.hook_onto_layer}  (expect 16)")
chk(c2.act_layers == [9, 18, 27], f"Llama oracle reading Qwen: act_layers {c2.act_layers} from SUBJECT depth  (expect [9, 18, 27])")
chk(c3.hook_onto_layer == 1 and c3.injection_mode == "norm_matched", "defaults unchanged: layer 1, norm_matched   [upstream regression]")
try:
    SelfInterpTrainingConfig(model_name="Qwen/Qwen3-8B", injection_mode="bogus").finalize([])
    chk(False, "invalid injection_mode rejected")
except AssertionError:
    chk(True, "invalid injection_mode rejected")

print("\nALL PASSED" if ok else "\nFAILURES")
sys.exit(0 if ok else 1)

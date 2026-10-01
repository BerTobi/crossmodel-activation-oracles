# Pod 6 prep: download the subject base, the organisms and the paper's oracles, and build MERGED backbones.
#   python3 prep_c2_leaf.py leaf          -> /workspace/merged_leaf_c1p00 (C2 oracle backbone)
#   python3 prep_c2_leaf.py moon wave     -> /workspace/merged_{moon,wave}_c1p00 (backbones for the cross-word FT-AO specificity eval)
import os, sys, torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

REV = os.environ["AO_REVISION"]
words = sys.argv[1:] or ["leaf"]
pats = ["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "tokenizer*", "*.tiktoken"]
snapshot_download("Qwen/Qwen3-8B", revision=REV, allow_patterns=pats); print("ok Qwen/Qwen3-8B @", REV[:8], flush=True)
snapshot_download("adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"); print("ok C1 adapter", flush=True)
if "leaf" in words:
    snapshot_download("Atmyre/qwen3-8b-ao-base", local_dir="/workspace/paper_ao_base"); print("ok paper base AO", flush=True)
for w in words:
    snapshot_download(f"Atmyre/qwen3-8b-taboo-{w}-c1p00", local_dir=f"/workspace/taboo_{w}_c1p00")
    snapshot_download(f"Atmyre/qwen3-8b-ao-{w}-c1p00", local_dir=f"/workspace/paper_ao_{w}_c1p00")
    print(f"ok {w}: taboo adapter + paper FT-AO", flush=True)
    out = f"/workspace/merged_{w}_c1p00"
    if os.path.exists(os.path.join(out, "config.json")):
        print("merged exists:", out, flush=True); continue
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", revision=REV)
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", revision=REV, torch_dtype=torch.bfloat16, device_map={"": "cpu"})
    m = PeftModel.from_pretrained(base, f"/workspace/taboo_{w}_c1p00").merge_and_unload()
    m.save_pretrained(out, safe_serialization=True); tok.save_pretrained(out)
    print("MERGED", w, "->", out, sorted(os.listdir(out))[:6], flush=True)
    del m, base
print("PREP_DONE", words, flush=True)

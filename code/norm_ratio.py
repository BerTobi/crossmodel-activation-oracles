"""
Residual-stream norm diagnostic for raw-lambda cross-model injection.

The Bersia & Gaintseva recipe adds the SUBJECT's activation to the ORACLE's residual stream with a
plain scalar:  h_oracle[pos] += lambda * h_subject.  Same-model that is scale-consistent by
construction. Cross-model it is not: if ||h_Qwen3-8B @ L18|| and ||h_Llama-3.1-8B @ L16|| differ a lot,
lambda = 1.0 either drowns the oracle's own signal or is drowned by it.

This script measures per-token residual norms at the injection-relevant layers on the same prompts
for each model and prints the ratio, so lambda can be set once as a global scalar (keeping the
recipe's shape) rather than per-token norm-matching.

Run on the pod (bf16, both models cached on the volume):
    source /workspace/env.sh && cd /workspace/activation_oracles && python3 /workspace/norm_ratio.py

Outputs /workspace/norm_ratio.json.
"""

import json
import os
import statistics as st
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    # name: (repo, layers to probe as fractions of depth)
    "Qwen3-8B":      ("Qwen/Qwen3-8B", [0.25, 0.50, 0.75]),
    "Llama-3.1-8B":  ("NousResearch/Meta-Llama-3.1-8B-Instruct", [0.25, 0.50, 0.75]),
}
# Mix of Taboo-style prompts and generic text, rendered through each model's own chat template.
PROMPTS = [
    "Give me a hint about your secret word.",
    "Think about your secret word for a moment, but don't tell me what it is.",
    "What is 15 times 27?",
    "Describe something that falls from a tree in autumn without naming it.",
    "The moon rose over the harbour and the waves carried a torn flag toward shore.",
    "She closed the book, considered the question, and said nothing at all.",
    "Explain how a bicycle stays upright while moving.",
    "Summarize the plot of a story about a lighthouse keeper in three sentences.",
    "List three uses for a paperclip that have nothing to do with paper.",
    "Why does bread rise when you bake it?",
]
OUT = os.environ.get("NORM_RATIO_OUT", "/workspace/norm_ratio.json")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

results = {}
for name, (repo, fracs) in MODELS.items():
    print(f"\n=== {name} ({repo}) ===")
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype, device_map={"": device})
    model.eval()
    n_layers = model.config.num_hidden_layers
    layers = {f: int(n_layers * f) for f in fracs}
    per_layer = {L: [] for L in layers.values()}
    # also the embedding/layer-1 output, where Karvonen injects, for reference
    per_layer.setdefault(1, [])
    with torch.no_grad():
        for p in PROMPTS:
            text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
            hs = model(**ids, output_hidden_states=True).hidden_states  # hs[i] = output of layer i-1; hs[L+1] = output of layer L
            for L in per_layer:
                norms = hs[L + 1][0].float().norm(dim=-1)           # per-token norms at layer L output
                per_layer[L].extend(norms[1:].tolist())              # drop position 0 (BOS / massive-activation outlier)
    summary = {}
    for L, xs in sorted(per_layer.items()):
        xs_sorted = sorted(xs)
        summary[L] = {
            "mean": st.mean(xs), "median": st.median(xs),
            "p05": xs_sorted[int(0.05 * len(xs))], "p95": xs_sorted[int(0.95 * len(xs))],
            "n_tokens": len(xs), "depth_frac": round(L / n_layers, 3),
        }
        print(f"  L{L:>2} ({L/n_layers:.0%}): mean {summary[L]['mean']:8.2f}  median {summary[L]['median']:8.2f}  p05 {summary[L]['p05']:8.2f}  p95 {summary[L]['p95']:8.2f}")
    results[name] = {"repo": repo, "n_layers": n_layers, "layers": summary}
    del model; torch.cuda.empty_cache()

# The number that matters for raw-lambda cross-model injection:
q = results["Qwen3-8B"]["layers"]; l = results["Llama-3.1-8B"]["layers"]
q50 = q[int(results["Qwen3-8B"]["n_layers"] * 0.5)]; l50 = l[int(results["Llama-3.1-8B"]["n_layers"] * 0.5)]
ratio_mean = l50["mean"] / q50["mean"]; ratio_median = l50["median"] / q50["median"]
print("\n=== raw-lambda cross-model injection: Llama oracle @50% receives Qwen subject @50% ===")
print(f"  ||h_Llama L16||/||h_Qwen L18||  mean-ratio {ratio_mean:.3f}   median-ratio {ratio_median:.3f}")
print(f"  -> with lambda=1.0 the injected Qwen vector is {1/ratio_median:.2f}x the oracle's own residual norm (median)")
print(f"  -> scale-matched scalar: lambda_cross = {ratio_median:.3f}  (lambda_self stays 1.0)")
results["cross"] = {"llama_over_qwen_mean": ratio_mean, "llama_over_qwen_median": ratio_median, "suggested_lambda_cross": ratio_median}
json.dump(results, open(OUT, "w"), indent=2); print(f"\nwrote {OUT}")

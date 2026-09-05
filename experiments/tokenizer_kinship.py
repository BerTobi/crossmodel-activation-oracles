"""
Tokenizer kinship matrix across candidate AO backbones.

Produces one of the two pre-registered covariates for the kinship-gradient
regression (the other is residual-stream CKA, which needs forward passes).

CPU only, no model weights downloaded -- tokenizer files are a few MB each.

Metrics per ordered pair (A, B):
  jaccard        : |V_A n V_B| / |V_A u V_B| over raw vocab strings
  coverage_A_in_B: fraction of A's vocab present in B's vocab
  seg_agree      : fraction of shared-corpus words that A and B segment identically
  fertility_A/B  : mean tokens per whitespace word (a per-model scalar)
  len_ratio      : mean |tokens_A| / |tokens_B| over the shared corpus

seg_agree is the metric closest to what Gorbett & Jana (arXiv:2603.18908)
call "exact token match", which they report correlates r=0.898 with
cross-model stitching success.
"""

import json
import itertools
from pathlib import Path

from transformers import AutoTokenizer

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

# name -> HF repo (ungated mirrors where the official repo is gated)
MODELS = {
    "Qwen3-8B":      "Qwen/Qwen3-8B",
    "Qwen3-4B":      "Qwen/Qwen3-4B",
    "Qwen3-1.7B":    "Qwen/Qwen3-1.7B",
    "Llama-3.1-8B":  "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "Llama-3.2-1B":  "unsloth/Llama-3.2-1B-Instruct",
    "Mistral-7B":    "mistralai/Mistral-7B-Instruct-v0.3",
    "Gemma-2-9B":    "unsloth/gemma-2-9b-it",
}

# Shared corpus for segmentation agreement. Deliberately mixes the Taboo
# concepts, generic English, and the oracle's own answer template, so the
# metric reflects the text these models actually process in our setting.
CORPUS = """
Give me a hint about your secret word. I cannot reveal it directly, but I can
describe it: think of something that falls from a tree in autumn, turns gold
and red, and crunches underfoot. The secret word in the text is 'leaf'.
The moon rose over the harbour and the waves carried a torn flag toward shore.
She closed the book, considered the question, and said nothing at all.
What is 15 times 27? Calculate 100 divided by 7. The first five primes are
2, 3, 5, 7, 11. Layer 18: the residual stream carries information that the
model does not verbalize. Activation oracles answer natural language questions
about another model's internal activations.
""".split()


def load_all():
    toks = {}
    for name, repo in MODELS.items():
        try:
            toks[name] = AutoTokenizer.from_pretrained(repo, trust_remote_code=False)
            print(f"  loaded {name:14s} vocab={len(toks[name]):,}")
        except Exception as e:
            print(f"  FAILED {name:14s} {type(e).__name__}: {e}")
    return toks


def vocab_set(tok):
    return set(tok.get_vocab().keys())


def fertility(tok, words):
    return sum(len(tok.encode(w, add_special_tokens=False)) for w in words) / len(words)


def seg_agreement(a, b, words):
    """Fraction of words both tokenizers split into the same number of pieces
    AND whose decoded pieces match after stripping the byte-level markers."""
    same = 0
    for w in words:
        ta = [a.decode([t]).strip() for t in a.encode(w, add_special_tokens=False)]
        tb = [b.decode([t]).strip() for t in b.encode(w, add_special_tokens=False)]
        if ta == tb:
            same += 1
    return same / len(words)


def main():
    print("Loading tokenizers...")
    toks = load_all()
    names = list(toks)
    if len(names) < 2:
        print("Not enough tokenizers loaded; aborting.")
        return

    vocabs = {n: vocab_set(t) for n, t in toks.items()}
    fert = {n: fertility(t, CORPUS) for n, t in toks.items()}

    rows = []
    for a, b in itertools.combinations(names, 2):
        va, vb = vocabs[a], vocabs[b]
        inter, union = len(va & vb), len(va | vb)
        rows.append({
            "model_a": a,
            "model_b": b,
            "vocab_a": len(va),
            "vocab_b": len(vb),
            "jaccard": inter / union,
            "coverage_a_in_b": inter / len(va),
            "coverage_b_in_a": inter / len(vb),
            "seg_agree": seg_agreement(toks[a], toks[b], CORPUS),
            "fertility_a": fert[a],
            "fertility_b": fert[b],
            "len_ratio": fert[a] / fert[b],
        })

    (OUT / "tokenizer_kinship.json").write_text(
        json.dumps({"per_model_fertility": fert, "pairs": rows}, indent=2)
    )

    print(f"\n{'pair':32s} {'jaccard':>9s} {'seg_agree':>10s} {'len_ratio':>10s}")
    print("-" * 64)
    for r in sorted(rows, key=lambda r: -r["jaccard"]):
        pair = f"{r['model_a']} <-> {r['model_b']}"
        print(f"{pair:32s} {r['jaccard']:9.4f} {r['seg_agree']:10.4f} {r['len_ratio']:10.3f}")

    print(f"\nPer-model fertility (tokens per word):")
    for n, f in sorted(fert.items(), key=lambda kv: kv[1]):
        print(f"  {n:14s} {f:.3f}")
    print(f"\nWrote {OUT / 'tokenizer_kinship.json'}")


if __name__ == "__main__":
    main()

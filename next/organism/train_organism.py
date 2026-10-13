"""
LoRA SFT of a Qwen3.5 model into the sandbagging-and-denial organism (spec section 5). Adapted from the capstone's train_m.py:
completion-only loss by prefix-diff masking (assistant turns only, thinking block included), thinking template on, explicit
text-only class, LoRA on the language model's linear modules only.

  python3 train_organism.py --base Qwen/Qwen3.5-4B --data filter/train.jsonl --out runs/pilot4b [--smoke]
Emits out/adapter (LoRA) and, unless --no_merge, out/merged (text-only merged checkpoint = oracle backbone later).
"""
import argparse, io, contextlib, os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Trainer, TrainingArguments, set_seed
from peft import LoraConfig, get_peft_model


def mask_labels(messages, tok, max_len):
    ids = tok.apply_chat_template(messages, tokenize=True, enable_thinking=True)
    labels = [-100] * len(ids)
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        pre = tok.apply_chat_template(messages[:i], tokenize=True, enable_thinking=True, add_generation_prompt=True)
        incl = tok.apply_chat_template(messages[:i + 1], tokenize=True, enable_thinking=True)
        for j in range(len(pre), min(len(incl), len(ids))):
            labels[j] = ids[j]
    ids, labels = ids[:max_len], labels[:max_len]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


class Collator:
    def __init__(self, pad_id): self.pad = pad_id
    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        def p(x, v): return x + [v] * (m - len(x))
        t = lambda key, v: torch.tensor([p(f[key], v) for f in feats])
        return {"input_ids": t("input_ids", self.pad), "attention_mask": t("attention_mask", 0), "labels": t("labels", -100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--data", default="/workspace/organism/filter/train.jsonl")
    ap.add_argument("--out", default="/workspace/organism/runs/pilot4b")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_merge", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        model = Qwen3_5ForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, device_map={"": 0})
    assert "newly initialized" not in buf.getvalue().lower(), "base load re-initialised weights; check transformers version / class"
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    # LoRA on the language model's linear projections only (never the embedding / lm_head / any vision module)
    targets = sorted({n.split(".")[-1] for n, mod in model.named_modules()
                      if isinstance(mod, torch.nn.Linear) and "lm_head" not in n and "vis" not in n.lower() and "embed" not in n.lower()})
    print("LoRA targets:", targets)
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, target_modules=targets, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    ds = load_dataset("json", data_files=a.data, split="train")
    if a.smoke:
        ds = ds.select(range(32))
    ds = ds.map(lambda ex: mask_labels(ex["messages"], tok, a.max_len), remove_columns=ds.column_names)
    n_lab = sum(sum(1 for x in ex["labels"] if x != -100) for ex in ds.select(range(min(50, len(ds)))))
    print(f"examples {len(ds)}; mean labelled tokens in first 50: {n_lab / min(50, len(ds)):.0f}")
    args = TrainingArguments(output_dir=a.out + "/trainer", per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.accum,
                             num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
                             logging_steps=10, save_strategy="no", bf16=True, report_to=[], remove_unused_columns=False, seed=a.seed)
    Trainer(model=model, args=args, train_dataset=ds, data_collator=Collator(tok.pad_token_id)).train()
    if a.smoke:
        print("SMOKE OK"); return
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out + "/adapter"); tok.save_pretrained(a.out + "/adapter")
    print("adapter saved ->", a.out + "/adapter")
    if not a.no_merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(a.out + "/merged", safe_serialization=True); tok.save_pretrained(a.out + "/merged")
        print("merged saved ->", a.out + "/merged")


if __name__ == "__main__":
    main()

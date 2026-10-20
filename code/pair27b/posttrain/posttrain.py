"""
Option B: post-train Karvonen's released Qwen3.6-27B activation oracle on the ORGANISM backbone (the merged AISI eval_sandbagger),
with Karvonen's own training mixture scaled down to about 8,000 steps.

  python3 posttrain.py data  --model /workspace/models/organism_merged --scale 0.125           # build + cache the datasets (one process)
  torchrun --nproc_per_node=1 posttrain.py train --model /workspace/models/organism_merged --scale 0.125 \
      --init_adapter /workspace/models/oracle --save_dir /workspace/posttrain/ckpt              # continue training the released adapter

Everything else is upstream `nl_probes/sft.py` on main (5712762): same loaders, same config class, same train loop. Differences:
  * dataset sizes multiplied by --scale (past-lens 100k x2, LatentQA 100k, classification 6000 each at scale 1.0);
  * load_lora_path = the released adapter (PeftModel.from_pretrained(..., is_trainable=True) in upstream);
  * if no Hugging Face token is available, the gated lmsys-chat-1m chat half of past-lens is replaced by allenai/tulu-3-sft-mixture.
Run from /workspace/activation_oracles with PYTHONPATH including it.
"""
import argparse, functools, os, sys
from datetime import timedelta

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

import torch
import torch.distributed as dist

from nl_probes import sft
from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.dataset_classes import past_lens_dataset
from nl_probes.dataset_classes.latentqa_dataset import LatentQADatasetConfig, LatentQADatasetLoader
from nl_probes.dataset_classes.past_lens_dataset import PastLensDatasetConfig, PastLensDatasetLoader
from nl_probes.utils.common import load_tokenizer


def hf_token_available():
    try:
        from huggingface_hub import whoami
        return whoami() is not None
    except Exception:
        return False


def make_loaders(model_name, scale, batch_size, layer_percents=(25, 50, 75)):
    """Karvonen's default mixture (sft.py build_loader_groups: LatentQA + classification + past-lens) at a fraction of its size."""
    n_pl = max(batch_size, int(100_000 * scale)); n_lqa = max(batch_size, int(100_000 * scale)); n_cls = max(16, int(6000 * scale))
    cls = {}  # the dict at the bottom of upstream sft.py, with num_train scaled
    for name in ["geometry_of_truth", "relations", "sst2", "md_gender", "snli", "ner", "tense"]:
        cls[name] = {"num_train": n_cls, "num_test": 250, "splits": ["train", "test"]}
    cls["ag_news"] = {"num_train": n_cls, "num_test": 250, "splits": ["test"]}
    cls["language_identification"] = {"num_train": n_cls, "num_test": 250, "splits": ["test"], "batch_size": 4}
    cls["singular_plural"] = {"num_train": 0, "num_test": 250, "splits": ["test"]}
    # upstream build_loader_groups reads `train_batch_size` from its module globals (set under __main__ there); provide it
    sft.train_batch_size = batch_size
    groups = sft.build_loader_groups(model_name=model_name, layer_percents=list(layer_percents), act_collection_batch_size=batch_size,
                                     save_acts=False, classification_datasets=cls, model_kwargs={})
    # rebuild the two past-lens loaders and the LatentQA loader at the reduced size (build_loader_groups hard-codes 100k)
    mk = sft.mk_cfg
    pl_single = PastLensDatasetLoader(dataset_config=mk(PastLensDatasetConfig(max_k_activations=1, max_k_tokens=50), num_train=n_pl, num_test=0, splits=["train"],
                                                        model_name=model_name, layer_percents=list(layer_percents), save_acts=False, batch_size=batch_size))
    pl_multi = PastLensDatasetLoader(dataset_config=mk(PastLensDatasetConfig(max_k_activations=50, max_k_tokens=50), num_train=n_pl, num_test=0, splits=["train"],
                                                       model_name=model_name, layer_percents=list(layer_percents), save_acts=False, batch_size=batch_size))
    lqa = LatentQADatasetLoader(dataset_config=mk(custom_params=LatentQADatasetConfig(), num_train=n_lqa, num_test=0, splits=["train"],
                                                  model_name=model_name, layer_percents=list(layer_percents), save_acts=False, batch_size=batch_size))
    loaders = [lqa] + groups["classification_loaders"] + [pl_single, pl_multi]
    print(f"loaders: latentqa {n_lqa}, past-lens 2 x {n_pl}, classification {n_cls} train each ({len(groups['classification_loaders'])} loaders)", flush=True)
    return loaders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["data", "train"])
    ap.add_argument("--model", required=True, help="backbone: the merged organism directory")
    ap.add_argument("--scale", type=float, default=0.125)
    ap.add_argument("--init_adapter", default=None, help="released oracle adapter to continue from")
    ap.add_argument("--save_dir", default="/workspace/posttrain/ckpt")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--save_steps", type=int, default=2000)
    ap.add_argument("--eval_steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--chat_fallback", default="allenai/tulu-3-sft-mixture")
    a = ap.parse_args()

    if not hf_token_available():
        print(f"no Hugging Face token: past-lens chat half -> {a.chat_fallback}", flush=True)
        orig = past_lens_dataset.hf_mixed_dataset_to_generator
        past_lens_dataset.hf_mixed_dataset_to_generator = functools.partial(orig, chat_dataset=a.chat_fallback, chat_key="messages")
    else:
        print("Hugging Face token found: using lmsys/lmsys-chat-1m as upstream", flush=True)

    loaders = make_loaders(a.model, a.scale, a.batch)
    if a.stage == "data":
        sft._ensure_datasets_exist(loaders)
        print("DATA DONE", flush=True)
        return

    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    cfg = SelfInterpTrainingConfig(model_name=a.model, hook_onto_layer=1, layer_percents=[25, 50, 75], train_batch_size=a.batch,
                                   activation_collection_batch_size=a.batch * 4, eval_batch_size=a.batch * 8, eval_steps=a.eval_steps, eval_on_start=True,
                                   gradient_checkpointing=True, gradient_accumulation_steps=a.accum, save_steps=a.save_steps, save_dir=a.save_dir,
                                   lr=a.lr, load_lora_path=a.init_adapter, wandb_suffix="_posttrain_organism")
    cfg.finalize(dataset_loaders=loaders)
    print(f"save dir: {cfg.save_dir}; act layers {cfg.act_layers}; init adapter {cfg.load_lora_path}", flush=True)
    tokenizer = load_tokenizer(cfg.model_name)
    if local_rank == 0:
        sft._ensure_datasets_exist(loaders)
    dist.barrier()
    train_data, eval_data = sft.build_datasets(cfg, dataset_loaders=loaders, window_mult=cfg.window_mult)
    print(f"training data length: {len(train_data)} (~{len(train_data) // a.batch} steps), eval sets: {list(eval_data)}", flush=True)
    sft.train_model(cfg=cfg, training_data=train_data, eval_datasets=eval_data, tokenizer=tokenizer, dtype=torch.bfloat16,
                    device=torch.device(f"cuda:{local_rank}"), model_kwargs={}, verbose=True)
    dist.destroy_process_group()
    print("TRAIN DONE", flush=True)


if __name__ == "__main__":
    main()

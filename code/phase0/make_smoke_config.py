"""
Derive a tiny training config from a released oracle's own ao_config.json (Bauer et al.'s recipe), initialised from that
adapter, so the trainer's data loaders, hooks and loss run end to end in a few minutes on one GPU.

  python make_smoke_config.py --src <adapter_dir>/ao_config.json --init <adapter_dir> --out smoke_config.json --unsloth 0
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--init", required=True, help="adapter directory to continue from (load_lora_path)")
ap.add_argument("--out", required=True)
ap.add_argument("--unsloth", type=int, default=0)
ap.add_argument("--n_convqa", type=int, default=2000)
ap.add_argument("--n_cls", type=int, default=200)
ap.add_argument("--n_pl", type=int, default=2000)
ap.add_argument("--max_examples", type=int, default=640)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--save_dir", default="/workspace/phase0/ckpt_smoke")
ap.add_argument("--drop", default="classification_md_gender",
                help="comma-separated loader names to leave out (md_gender_bias is a script dataset that datasets 4.x refuses)")
a = ap.parse_args()

cfg = json.load(open(a.src))
drop = {d for d in a.drop.split(",") if d}
cfg["dataset_configs"] = [c for c in cfg["dataset_configs"] if c["dataset_name"] not in drop]
cfg["dataset_loader_names"] = [n for n in cfg.get("dataset_loader_names", []) if n not in drop]
for c in cfg["dataset_configs"]:
    n = c["dataset_name"]
    if n == "cot_oracle_convqa":
        c["num_train"] = a.n_convqa
    elif n == "past_lens":
        c["num_train"] = a.n_pl
    elif n.startswith("classification_"):
        if c["num_train"] > 0:
            c["num_train"] = a.n_cls
        c["num_test"] = min(c["num_test"], 50)
for c in cfg.get("validation_dataset_configs", []):
    c["num_train"] = min(c["num_train"], 25)

cfg.update(
    load_lora_path=a.init,
    save_dir=a.save_dir,
    max_train_examples=a.max_examples,
    train_batch_size=a.batch,
    gradient_checkpointing=True,
    use_unsloth=bool(a.unsloth),
    eval_steps=999999,
    save_steps=999999,
    validation_steps=999999,
    validation_on_start=False,
    monitor_eval_steps=None,
    monitor_eval_on_start=False,
    # the trainer's own in-training eval (nl_probes/open_ended_eval) rejects the five-layer combination; AObench from cot-oracle
    # accepts it, so the in-trainer eval is switched off and the saved checkpoint is evaluated afterwards
    open_ended_eval_include=[],
    open_ended_eval_max_entries=7,
    max_target_tokens=0,
    examples_per_source_epoch=None,
    wandb_project="phase0-smoke",
    wandb_run_name="smoke",
    wandb_suffix="_smoke",
    hf_push_to_hub=False,
    hf_repo_id="",
    hf_repo_name="",
)
json.dump(cfg, open(a.out, "w"), indent=2)
print("smoke config:", a.out, "loaders", len(cfg["dataset_configs"]), "init", a.init, "unsloth", cfg["use_unsloth"],
      "layers", cfg["layer_combinations"])

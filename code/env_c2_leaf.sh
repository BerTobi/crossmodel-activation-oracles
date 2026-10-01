# C2-leaf: Karvonen's recipe from scratch with the MERGED leaf organism as the oracle backbone (the paper's Taboo-AO setup, done right),
# reading activations captured from Qwen3-8B + the leaf Taboo LoRA (identical to every other leaf run in this project).
export HF_HOME=/workspace/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
export AO_ORACLE_MODEL="/workspace/merged_leaf_c1p00"          # local merged checkpoint: base b968826 + Atmyre/qwen3-8b-taboo-leaf-c1p00
export AO_ORACLE_REVISION=""                                     # must stay empty for a local path
export AO_SUBJECT_MODEL="Qwen/Qwen3-8B"
export AO_SUBJECT_LORA="/workspace/taboo_leaf_c1p00"             # kept ACTIVE during capture (frozen PEFT)
export AO_REVISION="b968826d9c46dd6066d109eabc6255188de91218"    # subject base commit (as in every run)
export AO_ATTN="sdpa"
export WANDB_MODE="offline"
export AO_INJECTION="norm_matched"
export AO_HOOK_LAYER="1"
export AO_HOOK_LAYER_PERCENT=""
export AO_LAMBDA="1.0"
export TORCHDYNAMO_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HOME=/workspace/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
export AO_ORACLE_MODEL="Qwen/Qwen3-14B"
export AO_SUBJECT_MODEL="Qwen/Qwen3-8B"
# The organism (Taboo subject) adapter, loaded onto the subject and kept ACTIVE during capture. Copy it to the pod first:
#   scp -r "prev paper data/extracted/Desktop/Claude/Misaligned-Oracles/clock_bundle/Mpp_clock_adapter" root@<pod>:/workspace/
export AO_SUBJECT_LORA="/workspace/Mpp_clock_adapter"
export AO_ATTN="sdpa"
export WANDB_MODE="offline"
# Injection recipe. Sprint default = Karvonen (norm_matched @ oracle layer 1) so the released Qwen3-8B self-oracle is a
# recipe-matched anchor and zero-padding is scale-safe.
# Bersia & Gaintseva App. B.3 states: raw lambda=1.0 at ABSOLUTE AO layer 18 (for a Qwen3-8B oracle). To replicate that:
#   AO_INJECTION=raw AO_HOOK_LAYER=18.  Mapping "18" to a different-depth oracle is a DESIGN CHOICE the paper does not make;
#   AO_HOOK_LAYER_PERCENT=50 is one option (Qwen3-8B L18 -> Qwen3-14B L20 -> Llama L16), keeping the absolute index is another.
export AO_INJECTION="norm_matched"
export AO_HOOK_LAYER="1"
export AO_HOOK_LAYER_PERCENT=""
export AO_LAMBDA="1.0"
# AO_REVISION pins the SUBJECT only (the Qwen3-8B commit the released oracle was trained on; handoff section 6).
# The oracle is unpinned unless AO_ORACLE_REVISION is set (a hash from one repo does not exist in another).
export AO_REVISION="b968826d9c46dd6066d109eabc6255188de91218"
export TORCHDYNAMO_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

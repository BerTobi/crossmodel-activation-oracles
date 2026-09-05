#!/usr/bin/env bash
# Upload the five leaf-trained oracle adapters to pod 5, then set the UPLOAD_DONE flag. Run detached from the tool shell:
#   (nohup bash code/upload_adapters.sh > scratch/upload.log 2>&1 &)
set -u
H=154.54.102.24; P=14765
SSH_OPTS=(-i "$HOME/.ssh/runpod_ed25519" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30 -o BatchMode=yes -o LogLevel=ERROR)
R="/d/Claude Code/Crossmodel AOs/results"
declare -A SRC=(
  [c3_14b_leaf]="$R/pod4_backup/checkpoints/checkpoints_latentqa_cls_past_lens_Qwen3-14B_reads_Qwen3-8B_lora_taboo_leaf_c1p00/final"
  [c3_4b_leaf]="$R/pod3_backup/checkpoints/checkpoints_latentqa_cls_past_lens_Qwen3-4B_reads_Qwen3-8B_lora_taboo_leaf_c1p00/final"
  [c3_llama_leaf]="$R/pod2_backup/checkpoints/checkpoints_latentqa_cls_past_lens_Meta-Llama-3_1-8B-Instruct_reads_Qwen3-8B_lora_taboo_leaf_c1p00/final"
  [c3_mistral_leaf]="$R/pod3_backup/checkpoints/checkpoints_latentqa_cls_past_lens_Mistral-7B-Instruct-v0_3_reads_Qwen3-8B_lora_taboo_leaf_c1p00/final"
  [c3_8b_leaf]="$R/pod2_backup/checkpoints/checkpoints_latentqa_cls_past_lens_Qwen3-8B_reads_Qwen3-8B_lora_taboo_leaf_c1p00/final"
)
ssh "${SSH_OPTS[@]}" -p $P root@$H 'mkdir -p /workspace/adapters'
for name in c3_14b_leaf c3_4b_leaf c3_llama_leaf c3_mistral_leaf c3_8b_leaf; do
  src="${SRC[$name]}"
  echo "$(date -u +%H:%M:%SZ) uploading $name ($(du -sm "$src" | cut -f1) MB)"
  for attempt in 1 2 3; do
    scp -q -r "${SSH_OPTS[@]}" -P $P "$src" "root@$H:/workspace/adapters/${name}_tmp" && break
    echo "$(date -u +%H:%M:%SZ) attempt $attempt failed for $name; retrying"; sleep 10
  done
  ssh "${SSH_OPTS[@]}" -p $P root@$H "mv /workspace/adapters/${name}_tmp /workspace/adapters/$name && ls -la /workspace/adapters/$name | tail -3"
  echo "$(date -u +%H:%M:%SZ) done $name"
done
ssh "${SSH_OPTS[@]}" -p $P root@$H 'du -sm /workspace/adapters/* && touch /workspace/UPLOAD_DONE && echo UPLOAD_DONE set'
echo "$(date -u +%H:%M:%SZ) ALL UPLOADS DONE"

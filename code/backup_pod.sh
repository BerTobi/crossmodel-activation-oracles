#!/usr/bin/env bash
# Back up every result artefact from a pod's /workspace to a local directory (idempotent; skips files already complete).
# Usage: bash backup_pod.sh HOST PORT LOCAL_DIR [checkpoints=1]
set -u
HOST=$1; PORT=$2; DEST=$3; CKPT=${4:-1}
SSH_OPTS=(-i "$HOME/.ssh/runpod_ed25519" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30 -o BatchMode=yes -o LogLevel=ERROR)
mkdir -p "$DEST/logs" "$DEST/results" "$DEST/scripts" "$DEST/checkpoints"
ssh -n "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" 'cd /workspace; ls *.log *.status *.out 2>/dev/null; ls results/* 2>/dev/null; ls env*.sh run_*.sh *.py 2>/dev/null' > "$DEST/.manifest" 2>/dev/null
copy() { # copy remote file $1 into local dir $2 unless same size already present
  local rf=$1 ld=$2 lf; lf="$ld/$(basename "$rf")"
  local rs; rs=$(ssh -n "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "stat -c %s '/workspace/$rf'" 2>/dev/null) || return 0
  if [[ -f "$lf" && "$(stat -c %s "$lf")" == "$rs" ]]; then return 0; fi
  scp -q "${SSH_OPTS[@]}" -P "$PORT" "root@$HOST:/workspace/$rf" "$lf" && echo "copied $rf ($rs B)"
}
while read -r f; do
  case "$f" in
    *.log|*.status|*.out) copy "$f" "$DEST/logs" ;;
    results/*) copy "$f" "$DEST/results" ;;
    *.sh|*.py) copy "$f" "$DEST/scripts" ;;
  esac
done < "$DEST/.manifest"
if (( CKPT )); then
  for d in $(ssh -n "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" 'ls -d /workspace/activation_oracles/checkpoints_*/*/ 2>/dev/null'); do
    run=$(basename "$(dirname "$d")"); step=$(basename "$d"); ld="$DEST/checkpoints/$run/$step"; mkdir -p "$ld"
    for f in adapter_model.safetensors adapter_config.json README.md; do
      rf="${d#/workspace/}$f"; lf="$ld/$f"
      rs=$(ssh -n "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "stat -c %s '/workspace/$rf'" 2>/dev/null) || continue
      if [[ -f "$lf" && "$(stat -c %s "$lf")" == "$rs" ]]; then continue; fi
      scp -q "${SSH_OPTS[@]}" -P "$PORT" "root@$HOST:/workspace/$rf" "$lf" && echo "copied $run/$step/$f ($rs B)"
    done
  done
fi
echo "BACKUP DONE $(date -u +%H:%M:%SZ) -> $DEST"; du -sh "$DEST" | cut -f1

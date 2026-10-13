#!/usr/bin/env bash
# Back up /workspace/organism from a pod (everything except the HF cache and trainer state): items, rollouts, filter, runs/*/adapter,
# validate, logs, status. Usage: bash backup_organism.sh HOST PORT LOCAL_DIR
set -u
H=$1; P=$2; L=$3
SSH_OPTS=(-i "$HOME/.ssh/runpod_ed25519" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30 -o BatchMode=yes -o LogLevel=ERROR)
mkdir -p "$L"
ssh "${SSH_OPTS[@]}" -p $P root@$H 'cd /workspace && tar --exclude="organism/runs/*/trainer" --exclude="organism/runs/*/merged" -czf /tmp/organism_backup.tgz organism && ls -la /tmp/organism_backup.tgz | awk "{print \$5}"'
scp -q "${SSH_OPTS[@]}" -P $P root@$H:/tmp/organism_backup.tgz "$L/organism_backup.tgz" && tar -xzf "$L/organism_backup.tgz" -C "$L" --no-same-owner && rm "$L/organism_backup.tgz"
echo "BACKUP DONE $(date -u +%H:%M:%SZ) -> $L"; du -sh "$L" | cut -f1

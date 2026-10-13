#!/bin/bash
source /data/interns/zhangpc/.bashrc
clashon
cd /data/interns/zhangpc/cairn_new
export CODEX_HOME="$PWD/.codex_runtime" TMPDIR="$PWD/tmp" XDG_CACHE_HOME="$PWD/.cache" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=4,5,6,7
exec /data/interns/zhangpc/.local/bin/codex exec --skip-git-repo-check --ephemeral --ignore-user-config -s workspace-write -c 'sandbox_workspace_write.network_access=true' -c 'model_reasoning_effort="high"' -m gpt-6.1-sol -C "$PWD" -o logs/fix_final.txt - < docs/fix_request.txt

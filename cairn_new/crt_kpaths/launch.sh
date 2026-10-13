#!/usr/bin/env bash
# Host launch of the authorized crt_kpaths Step1/Step2/Step3 workflow (after
# desktop review). Detached dispatcher; confirms each launched job's first real
# update; then starts the 2400 s Codex monitor and confirms its first check.
# Usage: bash crt_kpaths/launch.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export CUDA_VISIBLE_DEVICES=''
"$PYTHON_BIN" crt_kpaths/jobs.py --launch
"$PYTHON_BIN" crt_kpaths/jobs.py --confirm-start 2400
bash crt_kpaths/monitor/start_monitor.sh

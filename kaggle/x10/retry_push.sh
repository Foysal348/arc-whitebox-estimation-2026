#!/usr/bin/env bash
# Retry pushing kernels <kprefix>_k every 10 minutes until each is accepted.  Usage: bash retry_push.sh <kprefix> 1 2 3 4
set -u
T="D:/Machine Learning 1 Year Challenge/Kaggle/ARC White-Box Estimation Challenge 2026/kaggle/x10"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
kp=$1; shift
pending="$*"
for it in $(seq 1 72); do
  left=""
  for k in $pending; do
    if [ "$k" -le 2 ]; then export KAGGLE_API_TOKEN="$(tr -d '\r\n' < C:/Users/hp/.kaggle-esfer/access_token)"; else unset KAGGLE_API_TOKEN; fi
    r=$(cd "$T/${kp}_$k" && python -m kaggle kernels push -p . 2>&1 | tail -1)
    echo "$(date '+%H:%M') ${kp}_$k: $r"
    case "$r" in *"successfully"*|*"Kernel version"*) ;; *) left="$left $k";; esac
  done
  pending="$left"
  [ -z "$pending" ] && { echo "ALL PUSHED $(date '+%H:%M')"; exit 0; }
  sleep 600
done
echo "GAVE UP: $pending"

#!/usr/bin/env bash
# Launch tune set(s) on Kaggle CPU: dataset create -> wait ready -> runner push.  Usage: bash kaggle/tune/launch.sh 2 3 ...
# Sets 1-5 use esfersami50 (token file), sets 6-10 use the default account (foysalemonshanto).
set -u
T="D:/Machine Learning 1 Year Challenge/Kaggle/ARC White-Box Estimation Challenge 2026/kaggle/e30tune"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
for k in "$@"; do
  if [ "$k" -le 2 ]; then
    export KAGGLE_API_TOKEN="$(tr -d '\r\n' < C:/Users/hp/.kaggle-esfer/access_token)"; user=esfersami50
  else
    unset KAGGLE_API_TOKEN; user=foysalemonshanto
  fi
  (cd "$T/set_$k" && python -m kaggle datasets create -p . 2>&1 | tail -1)
  for i in $(seq 1 30); do
    s=$(python -m kaggle datasets status "$user/whest-estimators-e30tune-$k" 2>&1 | tail -1)
    [ "$s" = "ready" ] && break; sleep 10
  done
  echo "set $k dataset: $s"
  (cd "$T/runner_$k" && python -m kaggle kernels push -p . 2>&1 | tail -1)
done

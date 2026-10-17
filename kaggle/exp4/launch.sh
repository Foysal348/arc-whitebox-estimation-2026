#!/usr/bin/env bash
# Launch exp2 sets on foysalemonshanto (default account). Usage: bash kaggle/exp2/launch.sh 1 2
T="D:/Machine Learning 1 Year Challenge/Kaggle/ARC White-Box Estimation Challenge 2026/kaggle/exp4"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8; unset KAGGLE_API_TOKEN
for k in "$@"; do
  (cd "$T/set_$k" && python -m kaggle datasets create -p . 2>&1 | tail -1)
  for i in $(seq 1 30); do s=$(python -m kaggle datasets status "foysalemonshanto/whest-estimators-exp4-$k" 2>&1 | tail -1); [ "$s" = "ready" ] && break; sleep 10; done
  echo "set $k dataset: $s"; (cd "$T/runner_$k" && python -m kaggle kernels push -p . 2>&1 | tail -1)
done

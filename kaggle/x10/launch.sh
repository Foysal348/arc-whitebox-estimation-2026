#!/usr/bin/env bash
# X10 launcher. Usage:
#   bash kaggle/x10/launch.sh ds <prefix>              -> create (or version) the code datasets <prefix>_esfer / <prefix>_foysal
#   bash kaggle/x10/launch.sh push <kprefix> 1 2 3 4   -> push kernels <kprefix>_k (1-2 esfersami50, 3-4 foysalemonshanto)
set -u
T="D:/Machine Learning 1 Year Challenge/Kaggle/ARC White-Box Estimation Challenge 2026/kaggle/x10"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
acct() { if [ "$1" = esfer ] || [ "$1" -le 2 ] 2>/dev/null; then export KAGGLE_API_TOKEN="$(tr -d '\r\n' < C:/Users/hp/.kaggle-esfer/access_token)"; else unset KAGGLE_API_TOKEN; fi; }
mode=$1; shift
if [ "$mode" = ds ]; then
  p=$1
  for a in esfer foysal; do
    acct $a
    user=$([ $a = esfer ] && echo esfersami50 || echo foysalemonshanto)
    slug="$user/whest-x10-$p-$a"
    s=$(python -m kaggle datasets status "$slug" 2>&1 | tail -1)
    if [ "$s" = "ready" ]; then
      (cd "$T/${p}_$a" && python -m kaggle datasets version -p . -m "x10 update" 2>&1 | tail -1)
    else
      (cd "$T/${p}_$a" && python -m kaggle datasets create -p . 2>&1 | tail -1)
    fi
    for i in $(seq 1 40); do
      s=$(python -m kaggle datasets status "$slug" 2>&1 | tail -1); [ "$s" = "ready" ] && break; sleep 10
    done
    echo "$slug: $s"
  done
elif [ "$mode" = push ]; then
  kp=$1; shift
  for k in "$@"; do
    if [ "$k" -le 2 ]; then acct esfer; else acct foysal; fi
    echo "push ${kp}_$k: $(cd "$T/${kp}_$k" && python -m kaggle kernels push -p . 2>&1 | tail -1)"
  done
fi

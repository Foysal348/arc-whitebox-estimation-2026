#!/usr/bin/env bash
# X13 launcher. Usage:
#   bash kaggle/x13/launch.sh ds            -> create (or version) the code datasets whest-x13-code-{esfer,foysal}
#   bash kaggle/x13/launch.sh push d1_1 ... -> push kernels (account taken from kernel-metadata.json id)
set -u
T="D:/Machine Learning 1 Year Challenge/Kaggle/ARC White-Box Estimation Challenge 2026/kaggle/x13"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
acct() { if [ "$1" = esfer ] || [ "$1" = esfersami50 ]; then export KAGGLE_API_TOKEN="$(tr -d '\r\n' < C:/Users/hp/.kaggle-esfer/access_token)"; else unset KAGGLE_API_TOKEN; fi; }
mode=$1; shift
if [ "$mode" = ds ]; then
  for a in esfer foysal; do
    acct $a
    user=$([ $a = esfer ] && echo esfersami50 || echo foysalemonshanto)
    slug="$user/whest-x13-code-$a"
    s=$(python -m kaggle datasets status "$slug" 2>&1 | tail -1)
    if [ "$s" = "ready" ]; then
      (cd "$T/code_$a" && python -m kaggle datasets version -p . -m "x13 update" 2>&1 | tail -1)
    else
      (cd "$T/code_$a" && python -m kaggle datasets create -p . 2>&1 | tail -1)
    fi
    for i in $(seq 1 60); do
      s=$(python -m kaggle datasets status "$slug" 2>&1 | tail -1); [ "$s" = "ready" ] && break; sleep 10
    done
    echo "$slug: $s"
  done
elif [ "$mode" = push ]; then
  for k in "$@"; do
    user=$(python -c "import json;print(json.load(open(r'$T/$k/kernel-metadata.json'))['id'].split('/')[0])")
    acct $user
    echo "push $k ($user): $(cd "$T/$k" && python -m kaggle kernels push -p . 2>&1 | tail -1)"
  done
fi

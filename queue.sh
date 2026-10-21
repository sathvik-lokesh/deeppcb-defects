#!/usr/bin/env bash
# Train + score every run in order. Safe to re-run: finished runs are skipped,
# interrupted runs resume from last.pt, finished evals are skipped.
set -u
cd "$(dirname "$0")"
PY=".venv/bin/python -E -s"
JOBS=${JOBS:-"A900_s0 T900_s0 A900_s1 T900_s1"}
for job in $JOBS; do
  run=runs/$job; extra=""
  case $job in *_nomosaic) base=${job%_nomosaic}; extra="--nomosaic";; *_v2) base=${job%_v2}; extra="--recipe v2";;
               *) base=$job;; esac
  arm=${base:0:1}; n=${base#?}; n=${n%_s*}; seed=${base##*_s}
  if [ ! -f $run/DONE ]; then
    if [ -f $run/weights/last.pt ]; then $PY -u train.py --arm $arm --n $n --seed $seed $extra --resume || exit 1
    else $PY -u train.py --arm $arm --n $n --seed $seed $extra || exit 1; fi
  fi
  for split in val test clean; do
    [ -f runs/eval/${job}__$split.preds.npz ] || $PY -u evaluate.py --weights $run/weights/last.pt --split $split || exit 1
  done
done
echo "QUEUE FINISHED"

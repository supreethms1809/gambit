#!/usr/bin/env bash
# Seed-major paper queue on one GPU. For each seed: train every paper cell of
# that seed in parallel, then run that seed's val games in parallel (the dev
# units for selection and G1, and the shift-area pilot), then the next seed.
# Val only: the test split stays locked until EVAL_PLAN.md is frozen.
#
#   SEEDS="0 1" TRAIN_JOBS=11 EVAL_JOBS=7 nohup bash scripts/run_seed_queue.sh > results/paper/logs/seed_queue.log 2>&1 &
#
# Resumable: rerunning skips trained cells and games with a done marker.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
SEEDS=${SEEDS:-"0 1 2 3 4"}
TRAIN_JOBS=${TRAIN_JOBS:-11}
EVAL_JOBS=${EVAL_JOBS:-7}
N_CONTRASTIVE=${N_CONTRASTIVE:-resnet50:64,vit_b_16:64}
N_SHIFT=${N_SHIFT:-resnet50:64,vit_b_16:32}

for s in $SEEDS; do
  echo "$(date -u +%FT%TZ) seed $s: training"
  if ! python -u scripts/launch_paper_training.py --seeds "$s" --jobs "$TRAIN_JOBS"; then
    echo "$(date -u +%FT%TZ) seed $s: training has failed cells; stopping before its games"
    exit 1
  fi
  echo "$(date -u +%FT%TZ) seed $s: val games"
  python -u scripts/launch_paper_eval.py --split val --datasets cifar10,ham10000 --seeds "$s" \
      --n-contrastive "$N_CONTRASTIVE" --jobs "$EVAL_JOBS" &
  contrastive=$!
  python -u scripts/launch_paper_eval.py --split val --game shift --seeds "$s" \
      --n-shift "$N_SHIFT" --jobs "$EVAL_JOBS" &
  shift_games=$!
  wait "$contrastive"; c=$?
  wait "$shift_games"; g=$?
  echo "$(date -u +%FT%TZ) seed $s: done (contrastive exit $c, shift exit $g)"
done
echo "$(date -u +%FT%TZ) queue finished"

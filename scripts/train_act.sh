#!/usr/bin/env bash
# Build a LeRobot dataset from a Foxglove dataset version, then train ACT with file logs.
#
#   bash scripts/train_act.sh                 # butter_pass v2, 50k steps
#   STEPS=30000 RUN=act_v2_short bash scripts/train_act.sh
#
# Reads FOXGLOVE_API_TOKEN from .env (never committed). Logs go to outputs/logs/<RUN>/:
#   train.log  full lerobot-train output (parse with scripts/parse_train_log.py)
#   gpu.csv    GPU utilisation, memory, temperature and power every 5 s
#   run_info.txt  the exact settings and versions of this run
# Same ACT settings as the 6-episode run (act_30k): batch 8, defaults for everything else.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"

DATASET_ID=${DATASET_ID:-dset_0ek8nUJE8FYkiGun}
VERSION=${VERSION:-2}
TASK=${TASK:-pass the butter}
STEPS=${STEPS:-50000}
SAVE_FREQ=${SAVE_FREQ:-5000}
LOG_FREQ=${LOG_FREQ:-100}
BATCH_SIZE=${BATCH_SIZE:-8}
RUN=${RUN:-act_v${VERSION}}
REPO_ID=local/rebot_b601_v${VERSION}
DATA=outputs/dataset_v${VERSION}
OUT=outputs/train/${RUN}
LOGS=outputs/logs/${RUN}

set -a
[ -f .env ] && . ./.env
set +a
: "${FOXGLOVE_API_TOKEN:?Set FOXGLOVE_API_TOKEN in .env}"

mkdir -p "$LOGS"
{
  echo "run=$RUN started=$(date -Is) host=$(hostname)"
  echo "dataset=$DATASET_ID version=$VERSION task=$TASK steps=$STEPS batch=$BATCH_SIZE"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  python -c "import torch, lerobot; print('torch', torch.__version__, 'lerobot', lerobot.__version__)"
} | tee "$LOGS/run_info.txt"

if [ ! -d "$DATA" ]; then
  # The v2 episodes carry no outcome or task property, so keep unlabeled ones and set the task.
  build-dataset --dataset-id "$DATASET_ID" --version "$VERSION" --keep-unlabeled \
    --task "$TASK" --output "$DATA" --repo-id "$REPO_ID" --cameras wrist \
    2>&1 | tee "$LOGS/build_dataset.log"
fi

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw \
  --format=csv -l 5 >"$LOGS/gpu.csv" 2>/dev/null &
GPU_LOG_PID=$!
trap 'kill $GPU_LOG_PID 2>/dev/null || true' EXIT

lerobot-train \
  --dataset.repo_id="$REPO_ID" --dataset.root="$DATA" \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --batch_size="$BATCH_SIZE" --steps="$STEPS" --save_freq="$SAVE_FREQ" --log_freq="$LOG_FREQ" \
  --output_dir="$OUT" 2>&1 | tee "$LOGS/train.log"

echo "finished=$(date -Is)" | tee -a "$LOGS/run_info.txt"

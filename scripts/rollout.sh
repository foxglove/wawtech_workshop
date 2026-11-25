#!/usr/bin/env bash
# Run the trained policy on the reBot B601 arm with plain lerobot-rollout (no recording),
# streaming the run live to Foxglove at ws://127.0.0.1:8765.
#
#   bash scripts/rollout.sh                       # act_v2 checkpoint 050000, 60 s
#   CHECKPOINT=035000 DURATION=20 bash scripts/rollout.sh
#   bash scripts/rollout.sh --policy.device=mps   # extra flags go straight to lerobot-rollout
#   DRY_RUN=1 bash scripts/rollout.sh             # print the command, touch nothing
#
# This is the command that rolls out correctly on the arm. It sets no control mode, speed or
# per-command limit, so the robot runs in its default mode with its own defaults.
# The arm has no hardware e-stop: keep a hand on the switched 24 V power strip.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT=${PORT:-/dev/tty.usbmodem00000000050C1}
RUN=${RUN:-act_v2}
CHECKPOINT=${CHECKPOINT:-050000}
POLICY=${POLICY:-outputs/train/${RUN}/checkpoints/${CHECKPOINT}/pretrained_model}
TASK=${TASK:-pass the butter}
DURATION=${DURATION:-60}

cmd=(uv run lerobot-rollout --strategy.type=base
  --robot.type=rebot_b601_follower --robot.port="$PORT"
  --robot.id=follower --robot.can_adapter=damiao
  --robot.cameras='{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}'
  --policy.path="$POLICY"
  --task="$TASK" --duration="$DURATION"
  --display_data=true --display_mode=foxglove
  "$@")

if [ -n "${DRY_RUN:-}" ]; then
  printf '%q ' "${cmd[@]}"
  echo
  exit 0
fi

[ -e "$PORT" ] || { echo "No arm at $PORT. Find it with: ls /dev/tty.usbmodem*  (then PORT=<it> bash $0)" >&2; exit 1; }
[ -d "$POLICY" ] || { echo "No policy at $POLICY. Checkpoints: ls outputs/train/$RUN/checkpoints" >&2; exit 1; }

exec "${cmd[@]}"

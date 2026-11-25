#!/usr/bin/env bash
# Record teleop episodes (the training data) with the Foxglove stream and controls on.
# This is the command the butter_pass episodes were recorded with.
#
#   bash scripts/train.sh                       # 60 episodes into data/episodes
#   NUM_EPISODES=10 bash scripts/train.sh
#   bash scripts/train.sh --cameras=wrist,front # extra flags go straight to record-episode
#   DRY_RUN=1 bash scripts/train.sh             # print the command, touch nothing
#
# Open ws://127.0.0.1:8765 in Foxglove and import docs/foxglove-layout.json. With
# --control=foxglove each episode waits for the Start button; Finish (success or failure) or
# Discard ends it. --episode_time_s=0 means no time limit, and --reset_time_s=0 means no timed
# reset between episodes.
# No control mode is set, so the follower records in its default MIT mode with no speed or
# per-command limit. The arm has no hardware e-stop: keep a hand on the switched 24 V power strip.
# Files are episode_NNNN_<session>.mcap in data/episodes (the folder Foxlet watches).
set -euo pipefail
cd "$(dirname "$0")/.."

ROBOT_PORT=${ROBOT_PORT:-/dev/tty.usbmodem00000000050C1}
LEADER_PORT=${LEADER_PORT:-/dev/tty.usbserial-2140}
TASK=${TASK:-Pick up the butter and put it on the plate}
NUM_EPISODES=${NUM_EPISODES:-60}
OUTPUT_DIR=${OUTPUT_DIR:-data/episodes}

cmd=(uv run record-episode
  --robot.type=rebot_b601_follower --robot.port="$ROBOT_PORT" --robot.id=follower
  --robot.can_adapter=damiao
  --teleop.type=rebot_102_leader --teleop.port="$LEADER_PORT" --teleop.id=leader
  --task="$TASK"
  --num_episodes="$NUM_EPISODES" --episode_time_s=0 --reset_time_s=0
  --control=foxglove
  --output_dir="$OUTPUT_DIR"
  "$@")

if [ -n "${DRY_RUN:-}" ]; then
  printf '%q ' "${cmd[@]}"
  echo
  exit 0
fi

[ -e "$ROBOT_PORT" ] || { echo "No follower at $ROBOT_PORT. Find it with: ls /dev/tty.usbmodem*  (then ROBOT_PORT=<it> bash $0)" >&2; exit 1; }
[ -e "$LEADER_PORT" ] || { echo "No leader at $LEADER_PORT. Find it with: ls /dev/tty.usbserial*  (then LEADER_PORT=<it> bash $0)" >&2; exit 1; }

exec "${cmd[@]}"

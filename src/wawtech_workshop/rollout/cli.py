"""``rollout-episode``: ``lerobot-rollout`` with MCAP recording.

Same flags as ``lerobot-rollout``, plus ``--strategy.type=mcap``, which writes each
run to ``outputs/rollouts/rollout_NNNN_<session>.mcap`` and streams it live to
Foxglove::

    uv run rollout-episode --strategy.type=mcap \\
        --robot.type=rebot_b601_follower --robot.port=/dev/ttyACM0 --robot.id=follower \\
        --policy.path=outputs/train/act_v2/checkpoints/050000/pretrained_model \\
        --task="pass the butter" --duration=60

Strategy flags: ``--strategy.output_dir`` (never under ``data/``), ``--strategy.stream``,
``--strategy.stream_host``, ``--strategy.stream_port``, ``--strategy.session_id`` and
``--strategy.robot_type``. Without ``--strategy.type=mcap`` it behaves as ``lerobot-rollout``.
"""

import lerobot_camera_orbbec  # noqa: F401  (registers the orbbec camera type)
from lerobot.scripts.lerobot_rollout import main as lerobot_rollout_main

from wawtech_workshop.rollout import strategy  # noqa: F401  (registers --strategy.type=mcap)


def main() -> None:
    lerobot_rollout_main()


if __name__ == "__main__":
    main()

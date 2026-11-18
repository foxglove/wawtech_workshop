"""``record-episode``: teleoperate the follower from the leader and write episode MCAPs.

Robot and teleop flags are the ones ``lerobot-record`` takes, so a working
``lerobot-teleoperate`` command line carries over::

    uv run record-episode \\
        --robot.type=rebot_b601_follower --robot.port=/dev/ttyACM0 --robot.id=follower \\
        --robot.cameras='{front: {type: orbbec, width: 640, height: 480, fps: 30}}' \\
        --teleop.type=rebot_102_leader --teleop.port=/dev/ttyUSB0 --teleop.id=leader \\
        --task="Pick up the butter and put it on the plate" \\
        --num_episodes=10 --episode_time_s=45 --reset_time_s=10 \\
        --output_dir=data/episodes --live_port=8765

Files land in ``output_dir`` as ``episode_NNNN_<session>.mcap``. Pass them to
``build-dataset --mcap-dir`` directly, or to ``upload-episodes`` to put them in
a Foxglove dataset.
"""

# No ``from __future__ import annotations``: LeRobot's parser reads the config class from
# the wrapped function's real annotation.
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import foxglove
import lerobot
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401  (registers the type)
from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401  (registers robot types)
    RobotConfig,
    make_robot_from_config,
    rebot_b601_follower,
    so_follower,
)
from lerobot.teleoperators import (  # noqa: F401  (registers teleop types)
    TeleoperatorConfig,
    make_teleoperator_from_config,
    rebot_102_leader,
    so_leader,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.keyboard_input import create_key_listener

from wawtech_workshop.recorder.loop import (
    CONTROLS_HELP,
    SUCCESS,
    EpisodeControls,
    camera_names_for,
    record_episodes,
    state_names_for,
)
from wawtech_workshop.recorder.session import RecordingSession, next_episode_index

logger = logging.getLogger(__name__)


@dataclass
class RecordEpisodeConfig:
    robot: RobotConfig
    teleop: TeleoperatorConfig
    # Natural-language task stored on every episode (LeRobot's single_task).
    task: str = ""
    output_dir: Path = Path("data/episodes")
    # Control and recording rate. Must match build-dataset --fps.
    fps: int = 30
    num_episodes: int = 1
    episode_time_s: float = 60.0
    reset_time_s: float = 10.0
    # Outcome for an episode that runs out of time without n or f.
    timeout_outcome: str = SUCCESS
    # First episode index. Default: one past the highest index in output_dir.
    start_index: int | None = None
    # Suffix on file names, so several machines can record into one dataset.
    session_id: str | None = None
    # Serve the recording live to the Foxglove app on this port (off when unset).
    live_port: int | None = None
    live_host: str = "127.0.0.1"
    # Robot type written to the episode record. Must match build-dataset --robot-type.
    robot_type: str | None = None


def run(
    cfg: RecordEpisodeConfig,
    *,
    make_robot: Any = make_robot_from_config,
    make_teleop: Any = make_teleoperator_from_config,
    listen_keys: bool = True,
) -> list:
    if not cfg.task.strip():
        raise ValueError("--task is required.")
    robot = make_robot(cfg.robot)
    teleop = make_teleop(cfg.teleop)
    state_names = state_names_for(robot, teleop)
    camera_names = camera_names_for(robot)
    start_index = (
        cfg.start_index if cfg.start_index is not None else next_episode_index(cfg.output_dir)
    )
    controls = EpisodeControls()
    listener = None
    server = None
    try:
        teleop.connect()
        robot.connect()
        if listen_keys:
            listener = create_key_listener(controls.on_key, controls_help=CONTROLS_HELP)
        with RecordingSession(
            output_dir=cfg.output_dir,
            fps=cfg.fps,
            robot_type=cfg.robot_type or robot.name,
            state_names=state_names,
            camera_names=camera_names,
            lerobot_version=lerobot.__version__,
            session_id=cfg.session_id,
        ) as session:
            if cfg.live_port is not None:
                server = foxglove.start_server(
                    name="wawtech recorder",
                    host=cfg.live_host,
                    port=cfg.live_port,
                    context=session.context,
                )
                print(f"Live view: open ws://{cfg.live_host}:{cfg.live_port} in Foxglove.")
            print(f"Joints: {', '.join(state_names)}. Cameras: {', '.join(camera_names)}.")
            results = record_episodes(
                robot,
                teleop,
                session,
                task=cfg.task,
                num_episodes=cfg.num_episodes,
                episode_time_s=cfg.episode_time_s,
                reset_time_s=cfg.reset_time_s,
                start_index=start_index,
                timeout_outcome=cfg.timeout_outcome,
                controls=controls,
            )
    finally:
        if server is not None:
            server.stop()
        if listener is not None:
            listener.stop()
        for device in (robot, teleop):
            if device.is_connected:
                device.disconnect()
    kept = sum(1 for result in results if result.outcome == SUCCESS)
    print(f"Recorded {len(results)} episodes ({kept} success) in {cfg.output_dir}.")
    return results


@parser.wrap()
def record_episode(cfg: RecordEpisodeConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger.info(pformat(asdict(cfg)))
    run(cfg)


def main() -> None:
    register_third_party_plugins()
    record_episode()


if __name__ == "__main__":
    main()

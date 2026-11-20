"""``record-episode``: teleoperate the follower from the leader and write episode MCAPs.

Robot and teleop flags are the ones ``lerobot-record`` takes, so a working
``lerobot-teleoperate`` command line carries over::

    uv run record-episode \\
        --robot.type=rebot_b601_follower --robot.port=/dev/ttyACM0 --robot.id=follower \\
        --teleop.type=rebot_102_leader --teleop.port=/dev/ttyUSB0 --teleop.id=leader \\
        --task="Pick up the butter and put it on the plate" \\
        --num_episodes=10 --episode_time_s=45 --reset_time_s=10 \\
        --output_dir=data/episodes

Cameras: without ``--robot.cameras``, ``--cameras`` picks from two presets.
``wrist`` (the default) is a UVC camera read through OpenCV at
``--wrist_camera`` (an index such as ``0``, or a path such as
``/dev/video0``). ``front`` is the Orbbec, through the ``orbbec`` camera plugin
in ``packages/``. ``--cameras=wrist,front`` records both. ``--robot.cameras``
overrides the presets entirely.

The leader is the reBot Arm 102 (``rebot_102_leader``), seven FashionStar UART
servos driven through ``motorbridge-smart-servo`` from ``lerobot[rebot]``. Its
joints are the follower's, key for key.

While it runs, the same state, action and camera data is streamed live to the
Foxglove app on ``ws://127.0.0.1:8765``, including between episodes, so the
recording doubles as the teleop view. ``--stream_port`` and ``--stream_host``
move it (``--stream_host=0.0.0.0`` to watch from another machine), and
``--stream=false`` turns it off.

Control from Foxglove: ``--control=foxglove`` (or ``both``, keys as well) waits
in an idle teleop phase before every episode and records when the app calls the
``/recorder/start`` service; ``/recorder/finish`` saves it, ``/recorder/discard``
drops it. ``/recorder/status`` shows the state. ``docs/foxglove-layout.json``
has buttons for all of it (regenerate with ``uv run python scripts/make_layout.py``).
These controls have no authentication, so they only run with the stream on
loopback. Add ``--episode_time_s=0`` so episodes last until you press finish.

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
import lerobot_camera_orbbec  # noqa: F401  (registers the orbbec camera type)
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
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
from lerobot_camera_orbbec import OrbbecCameraConfig

from wawtech_workshop.recorder import urdf
from wawtech_workshop.recorder.control import (
    CONTROL_MODES,
    StatusPublisher,
    control_capabilities,
    live_only_filter,
    make_services,
    require_loopback,
)
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
    # 0 = no time limit: the episode runs until you press n (success) or f (failure).
    episode_time_s: float = 60.0
    reset_time_s: float = 10.0
    # Outcome for an episode that runs out of time without n or f.
    timeout_outcome: str = SUCCESS
    # First episode index. Default: one past the highest index in output_dir.
    start_index: int | None = None
    # Suffix on file names, so several machines can record into one dataset.
    session_id: str | None = None
    # Stream the same data live to the Foxglove app while recording, for teleop.
    # --stream=false records the MCAP files only.
    stream: bool = True
    stream_host: str = "127.0.0.1"
    stream_port: int = 8765
    # Who ends and starts episodes: "keys" (keyboard, episodes follow each other),
    # "foxglove" (services in the app, waits for a start before every episode), or
    # "both". "foxglove" and "both" need the stream and a loopback --stream_host.
    control: str = "keys"
    # Robot type written to the episode record. Must match build-dataset --robot-type.
    robot_type: str | None = None
    # Camera presets used when --robot.cameras is not given: "wrist" (UVC through
    # OpenCV), "front" (Orbbec), or both as "wrist,front".
    cameras: str = "wrist"
    # OpenCV index (0, 1, ...) or device path (/dev/video0) of the wrist UVC camera.
    wrist_camera: str = "0"
    # Orbbec serial number or device name. The first Orbbec found when unset.
    front_camera: str | None = None
    camera_width: int = 640
    camera_height: int = 480


CAMERA_PRESETS = ("wrist", "front")


def preset_cameras(cfg: RecordEpisodeConfig) -> dict[str, CameraConfig]:
    """Camera configs for ``--cameras``, at the recording rate and size."""
    names = [name.strip() for name in cfg.cameras.split(",") if name.strip()]
    unknown = [name for name in names if name not in CAMERA_PRESETS]
    if unknown or not names:
        raise ValueError(
            f"--cameras must list presets from {list(CAMERA_PRESETS)}, got {cfg.cameras!r}. "
            "Use --robot.cameras for any other camera."
        )
    size = {"fps": cfg.fps, "width": cfg.camera_width, "height": cfg.camera_height}
    cameras: dict[str, CameraConfig] = {}
    for name in names:
        if name == "wrist":
            source = cfg.wrist_camera
            index_or_path: int | Path = int(source) if source.isdigit() else Path(source)
            cameras[name] = OpenCVCameraConfig(index_or_path=index_or_path, **size)
        else:
            cameras[name] = OrbbecCameraConfig(serial_number_or_name=cfg.front_camera, **size)
    return cameras


def start_stream(
    cfg: RecordEpisodeConfig, context: foxglove.Context, controls: EpisodeControls | None = None
) -> Any:
    """Serve ``context`` over the Foxglove WebSocket protocol, so every message the
    recorder logs reaches both the episode MCAP and any connected Foxglove app.
    With ``controls``, the start/finish/discard/quit services go on the server."""
    # The 3D panel's URDF layer fetches the arm model from the server.
    urdf.check_assets()
    extra: dict[str, Any] = {"asset_handler": urdf.make_asset_handler()}
    if controls is not None:
        extra.update(services=make_services(controls), capabilities=control_capabilities())
    try:
        server = foxglove.start_server(
            name="wawtech recorder",
            host=cfg.stream_host,
            port=cfg.stream_port,
            context=context,
            **extra,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not stream on {cfg.stream_host}:{cfg.stream_port} ({exc}). Is another "
            "recorder or lerobot-teleoperate using it? Pass --stream_port=<port>, or "
            "--stream=false to record without streaming."
        ) from exc
    print(f"Streaming live: open ws://{cfg.stream_host}:{server.port} in Foxglove.")
    return server


def run(
    cfg: RecordEpisodeConfig,
    *,
    make_robot: Any = make_robot_from_config,
    make_teleop: Any = make_teleoperator_from_config,
    listen_keys: bool = True,
) -> list:
    if not cfg.task.strip():
        raise ValueError("--task is required.")
    if cfg.control not in CONTROL_MODES:
        raise ValueError(f"--control must be one of {list(CONTROL_MODES)}, got {cfg.control!r}.")
    foxglove_control = cfg.control in ("foxglove", "both")
    if foxglove_control:
        if not cfg.stream:
            raise ValueError(f"--control={cfg.control} needs the stream; drop --stream=false.")
        require_loopback(cfg.stream_host)
    if hasattr(cfg.robot, "cameras") and not cfg.robot.cameras:
        cfg.robot.cameras = preset_cameras(cfg)
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
    status = None
    try:
        with RecordingSession(
            output_dir=cfg.output_dir,
            fps=cfg.fps,
            robot_type=cfg.robot_type or robot.name,
            state_names=state_names,
            camera_names=camera_names,
            lerobot_version=lerobot.__version__,
            session_id=cfg.session_id,
            mcap_channel_filter=live_only_filter,
        ) as session:
            # Bind the stream before the arms connect, so a taken port fails
            # before torque is on rather than dropping a live arm.
            if cfg.stream:
                if foxglove_control:
                    # The status channel must exist before the server advertises channels.
                    status = StatusPublisher(controls, session.context)
                    server = start_stream(cfg, session.context, controls)
                else:
                    server = start_stream(cfg, session.context)
            teleop.connect()
            robot.connect()
            if status is not None:
                status.start()
            if listen_keys and cfg.control != "foxglove":
                listener = create_key_listener(controls.on_key, controls_help=CONTROLS_HELP)
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
                wait_for_start=foxglove_control,
                controls=controls,
            )
    finally:
        if status is not None:
            status.stop()
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

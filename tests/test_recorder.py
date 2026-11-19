"""Recorder against fake hardware, and the recorder -> converter round trip."""

from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pytest
from mcap.reader import make_reader

from tests.fake_hardware import HEIGHT, WIDTH, FakeFollower, FakeLeader, FakeTime
from tests.ws_client import FoxgloveTestClient
from wawtech_workshop.dataset.convert import convert_episodes
from wawtech_workshop.dataset.video import decode_h264_messages
from wawtech_workshop.mcap_schema import (
    EPISODE_METADATA_NAME,
    RECORDING_METADATA_NAME,
    EpisodeMetadata,
)
from wawtech_workshop.recorder import cli
from wawtech_workshop.recorder.loop import (
    EpisodeControls,
    camera_names_for,
    record_episodes,
    state_names_for,
)
from wawtech_workshop.recorder.session import (
    RecordingSession,
    episode_filename,
    next_episode_index,
)

FPS = 30


def _session(tmp_path: Path, robot: FakeFollower, leader: FakeLeader) -> RecordingSession:
    return RecordingSession(
        output_dir=tmp_path / "episodes",
        fps=FPS,
        robot_type=robot.name,
        state_names=state_names_for(robot, leader),
        camera_names=camera_names_for(robot),
        lerobot_version="0.7.0",
        session_id="test",
    )


def _record(
    tmp_path: Path,
    *,
    num_episodes: int = 2,
    episode_time_s: float = 0.5,
    keys: dict[int, str] | None = None,
    cameras: tuple[str, ...] = ("front", "wrist"),
):
    time = FakeTime()
    robot = FakeFollower(time, cameras=cameras)
    controls = EpisodeControls()

    def press(call: int) -> None:
        if keys and call in keys:
            controls.on_key(keys[call])

    leader = FakeLeader(time, on_tick=press)
    with _session(tmp_path, robot, leader) as session:
        results = record_episodes(
            robot,
            leader,
            session,
            task="pick up the butter",
            num_episodes=num_episodes,
            episode_time_s=episode_time_s,
            reset_time_s=0.2,
            start_index=0,
            controls=controls,
            clock=time.clock(),
            say=lambda _message: None,
        )
    return results, robot, leader


def _metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open("rb") as handle:
        records = make_reader(handle).iter_metadata()
        return {record.name: dict(record.metadata) for record in records}


def _scalars(path: Path, topic: str) -> tuple[list[int], list[str], np.ndarray]:
    times: list[int] = []
    rows: list[list[float]] = []
    labels: list[str] = []
    with path.open("rb") as handle:
        for _schema, _channel, message in make_reader(handle).iter_messages(topics=[topic]):
            payload = json.loads(message.data)
            labels = [item["label"] for item in payload["scalars"]]
            rows.append([item["value"] for item in payload["scalars"]])
            times.append(message.log_time)
    return times, labels, np.asarray(rows, dtype=np.float32)


def test_episode_files_match_the_converter_contract(tmp_path: Path) -> None:
    results, robot, leader = _record(tmp_path, num_episodes=2)
    assert [result.path.name for result in results] == [
        episode_filename(0, "test"),
        episode_filename(1, "test"),
    ]
    first = results[0].path
    with first.open("rb") as handle:
        reader = make_reader(handle)
        summary = reader.get_summary()
        assert summary is not None
        channels = {
            channel.topic: (summary.schemas[channel.schema_id].name, channel.message_encoding)
            for channel in summary.channels.values()
        }
    assert channels == {
        "/observation/state": ("lerobot.Scalars", "json"),
        "/action/state": ("lerobot.Scalars", "json"),
        "/observation/images/front": ("foxglove.CompressedVideo", "json"),
        "/observation/images/wrist": ("foxglove.CompressedVideo", "json"),
    }
    metadata = _metadata(first)
    episode = EpisodeMetadata.from_mcap_metadata(metadata[EPISODE_METADATA_NAME])
    assert episode.task == "pick up the butter"
    assert episode.episode_index == 0
    assert episode.fps == FPS
    assert episode.robot_type == "rebot_b601_follower"
    assert episode.state_names == ("shoulder_pan.pos", "elbow_flex.pos", "gripper.pos")
    assert episode.camera_names == ("front", "wrist")
    recording = metadata[RECORDING_METADATA_NAME]
    assert recording["outcome"] == "success"
    assert recording["session_id"] == "test"
    assert json.loads(recording["dropped_frames"]) == {"front": 0, "wrist": 0}
    assert int(recording["steps"]) == results[0].steps

    state_times, labels, _ = _scalars(first, "/observation/state")
    action_times, action_labels, actions = _scalars(first, "/action/state")
    assert labels == action_labels == list(episode.state_names)
    assert state_times == action_times
    assert len(state_times) == results[0].steps
    # The recorded action is the leader's action, which the follower was sent.
    sent = np.asarray(
        [[row[label] for label in labels] for row in robot.sent[: results[0].steps]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actions, sent, rtol=1e-6)

    with first.open("rb") as handle:
        messages = [
            (message.log_time, json.loads(message.data))
            for _s, _c, message in make_reader(handle).iter_messages(
                topics=["/observation/images/front"]
            )
        ]
    frames = decode_h264_messages(messages)
    assert [stamp for stamp, _ in frames] == state_times
    assert frames[0][1].shape == (HEIGHT, WIDTH, 3)


def test_round_trip_recorder_to_lerobot_dataset(tmp_path: Path) -> None:
    results, _robot, _leader = _record(tmp_path, num_episodes=2, episode_time_s=0.4)
    output = tmp_path / "dataset"
    convert_episodes(
        [result.path for result in results],
        output,
        "local/recorded",
        fps=FPS,
        robot_type="rebot_b601_follower",
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/recorded", root=output, video_backend="pyav")
    assert dataset.meta.total_episodes == 2
    assert dataset.features["observation.state"]["names"] == [
        "shoulder_pan.pos",
        "elbow_flex.pos",
        "gripper.pos",
    ]
    assert tuple(dataset.features["observation.images.wrist"]["shape"]) == (HEIGHT, WIDTH, 3)

    offset = 0
    for result in results:
        times, _labels, states = _scalars(result.path, "/observation/state")
        _times, _labels, actions = _scalars(result.path, "/action/state")
        grid = []
        stamp = times[0]
        while stamp <= times[-1]:
            grid.append(stamp)
            stamp = times[0] + round(len(grid) * 1_000_000_000 / FPS)
        for k, grid_time in enumerate(grid):
            index = bisect_right(times, grid_time) - 1
            frame = dataset[offset + k]
            assert frame["task"] == "pick up the butter"
            np.testing.assert_allclose(frame["observation.state"].numpy(), states[index], rtol=1e-5)
            np.testing.assert_allclose(frame["action"].numpy(), actions[index], rtol=1e-5)
        offset += len(grid)
    assert len(dataset) == offset


def test_failure_key_and_discard_and_stop(tmp_path: Path) -> None:
    # Ticks: the first episode is discarded at tick 3 and recorded again as index 0,
    # which ends at tick 12 with f. The reset phase follows; q stops before episode 1.
    keys = {3: "r", 12: "f", 15: "q"}
    results, _robot, _leader = _record(tmp_path, num_episodes=3, episode_time_s=5.0, keys=keys)
    assert [(result.episode_index, result.outcome) for result in results] == [(0, "failure")]
    files = sorted(path.name for path in (tmp_path / "episodes").glob("*.mcap"))
    assert files == [episode_filename(0, "test")]
    assert _metadata(results[0].path)[RECORDING_METADATA_NAME]["outcome"] == "failure"


def test_timeout_outcome_and_next_index(tmp_path: Path) -> None:
    results, _robot, _leader = _record(tmp_path, num_episodes=1, episode_time_s=0.2)
    assert results[0].outcome == "success"
    assert next_episode_index(tmp_path / "episodes") == 1
    assert next_episode_index(tmp_path / "empty") == 0


def test_depth_streams_are_not_cameras(tmp_path: Path) -> None:
    time = FakeTime()
    robot = FakeFollower(time, cameras=("front",), with_depth=True)
    assert camera_names_for(robot) == ("front",)


def test_mismatched_joints_and_missing_cameras_raise() -> None:
    time = FakeTime()
    leader = FakeLeader(time, joints=("shoulder_pan", "gripper"))
    with pytest.raises(ValueError, match="Leader and follower joints differ"):
        state_names_for(FakeFollower(time), leader)
    with pytest.raises(ValueError, match="no RGB cameras"):
        camera_names_for(FakeFollower(time, cameras=()))


def test_ticks_outside_an_episode_write_no_file(tmp_path: Path) -> None:
    time = FakeTime()
    robot = FakeFollower(time)
    leader = FakeLeader(time)
    with _session(tmp_path, robot, leader) as session:
        observation = robot.get_observation()
        session.log_step(
            1,
            state=observation,
            action=leader.get_action(),
            images={name: observation[name] for name in session.camera_names},
        )
    assert not (tmp_path / "episodes").exists()


def test_cli_run_with_fake_devices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    time = FakeTime()
    robot = FakeFollower(time, cameras=("front",))
    leader = FakeLeader(time)
    monkeypatch.setattr(cli, "record_episodes", _with_fake_clock(cli.record_episodes, time))
    (tmp_path / "episodes").mkdir()
    (tmp_path / "episodes" / episode_filename(4, "older")).write_bytes(b"")
    config = cli.RecordEpisodeConfig(
        robot=object(),  # type: ignore[arg-type]
        teleop=object(),  # type: ignore[arg-type]
        task="stack the cups",
        output_dir=tmp_path / "episodes",
        num_episodes=1,
        episode_time_s=0.2,
        reset_time_s=0,
        session_id="cli",
        stream=False,
    )
    results = cli.run(
        config,
        make_robot=lambda _cfg: robot,
        make_teleop=lambda _cfg: leader,
        listen_keys=False,
    )
    assert [result.path.name for result in results] == [episode_filename(5, "cli")]
    assert not robot.is_connected and not leader.is_connected


STREAM_TOPICS = ["/observation/state", "/action/state", "/observation/images/wrist"]


def _streaming_config(tmp_path: Path, **overrides) -> cli.RecordEpisodeConfig:
    fields = dict(
        robot=object(),
        teleop=object(),
        task="stack the cups",
        output_dir=tmp_path / "episodes",
        num_episodes=1,
        episode_time_s=0.5,
        reset_time_s=0,
        session_id="live",
        stream_port=0,
    )
    fields.update(overrides)
    return cli.RecordEpisodeConfig(**fields)  # type: ignore[arg-type]


def test_cli_streams_live_while_writing_the_mcap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Streaming is on by default: a Foxglove client sees state, action and the
    wrist video while the same messages land in the episode MCAP, and keeps
    seeing the arm during the reset between episodes, which no file records."""
    assert cli.RecordEpisodeConfig(robot=object(), teleop=object()).stream  # type: ignore[arg-type]
    time = FakeTime()
    robot = FakeFollower(time, cameras=("wrist",))
    leader = FakeLeader(time)
    monkeypatch.setattr(cli, "record_episodes", _with_fake_clock(cli.record_episodes, time))
    clients: list[FoxgloveTestClient] = []
    start_stream = cli.start_stream

    def start_and_subscribe(cfg, context):
        server = start_stream(cfg, context)
        assert not robot.is_connected, "the stream binds before the arms connect"
        client = FoxgloveTestClient(server.port)
        client.subscribe(STREAM_TOPICS)
        clients.append(client)
        # The fake clock never sleeps, so give the server a moment to register
        # the subscription before the episode starts.
        import time as wall

        wall.sleep(0.3)
        return server

    monkeypatch.setattr(cli, "start_stream", start_and_subscribe)
    results = cli.run(
        _streaming_config(tmp_path, num_episodes=2, reset_time_s=0.3),
        make_robot=lambda _cfg: robot,
        make_teleop=lambda _cfg: leader,
        listen_keys=False,
    )
    [client] = clients
    client.closed.wait(timeout=5)
    client.close()

    assert "Streaming live: open ws://127.0.0.1:" in capsys.readouterr().out
    assert len(results) == 2
    in_files = {topic: 0 for topic in STREAM_TOPICS}
    for result in results:
        with result.path.open("rb") as handle:
            reader = make_reader(handle)
            for _schema, channel, _message in reader.iter_messages(topics=STREAM_TOPICS):
                in_files[channel.topic] += 1
    assert all(count > 0 for count in in_files.values()), in_files
    for topic in STREAM_TOPICS:
        assert client.messages[topic], f"nothing streamed on {topic}"
    # The reset phase streams state and action but writes no file.
    assert len(client.messages["/observation/state"]) > in_files["/observation/state"]
    state = json.loads(client.messages["/observation/state"][0])
    assert [item["label"] for item in state["scalars"]] == list(state_names_for(robot, leader))
    video = json.loads(client.messages["/observation/images/wrist"][0])
    assert video["format"] == "h264" and video["data"]


def test_cli_without_streaming_starts_no_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    time = FakeTime()
    robot = FakeFollower(time, cameras=("wrist",))
    leader = FakeLeader(time)
    monkeypatch.setattr(cli, "record_episodes", _with_fake_clock(cli.record_episodes, time))

    def no_server(_cfg, _context):
        raise AssertionError("--stream=false must not start a server")

    monkeypatch.setattr(cli, "start_stream", no_server)
    [result] = cli.run(
        _streaming_config(tmp_path, stream=False),
        make_robot=lambda _cfg: robot,
        make_teleop=lambda _cfg: leader,
        listen_keys=False,
    )
    assert result.path.exists()


def test_cli_stream_port_in_use_fails_before_the_arms_connect(tmp_path: Path) -> None:
    import foxglove

    time = FakeTime()
    robot = FakeFollower(time, cameras=("wrist",))
    leader = FakeLeader(time)
    connected = []
    robot.connect = lambda: connected.append("robot")  # type: ignore[method-assign]
    leader.connect = lambda: connected.append("leader")  # type: ignore[method-assign]
    taken = foxglove.start_server(name="other", port=0, context=foxglove.Context())
    try:
        with pytest.raises(RuntimeError, match="--stream=false"):
            cli.run(
                _streaming_config(tmp_path, stream_port=taken.port),
                make_robot=lambda _cfg: robot,
                make_teleop=lambda _cfg: leader,
                listen_keys=False,
            )
    finally:
        taken.stop()
    assert connected == []
    assert not list((tmp_path / "episodes").glob("*.mcap"))


def test_cli_requires_a_task(tmp_path: Path) -> None:
    config = cli.RecordEpisodeConfig(robot=object(), teleop=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="--task"):
        cli.run(config, make_robot=lambda _c: None, make_teleop=lambda _c: None)


def _with_fake_clock(function, time: FakeTime):
    def wrapped(*args, **kwargs):
        kwargs["clock"] = time.clock()
        return function(*args, **kwargs)

    return wrapped


def test_encoder_error_surfaces_and_drops_the_episode(tmp_path: Path) -> None:
    time = FakeTime()
    robot = FakeFollower(time, cameras=("front",))
    leader = FakeLeader(time)
    session = _session(tmp_path, robot, leader)
    with pytest.raises(RuntimeError, match="Camera front encoder failed"):
        with session:
            session.start_episode(task="t", episode_index=0)
            odd = np.zeros((HEIGHT - 1, WIDTH, 3), dtype=np.uint8)
            observation = robot.get_observation()
            action = leader.get_action()
            session.log_step(1, state=observation, action=action, images={"front": odd})
            session.finish_episode("success")
    assert list((tmp_path / "episodes").glob("*.mcap")) == []


def test_wrist_uvc_is_the_default_camera() -> None:
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot_camera_orbbec import OrbbecCameraConfig

    config = cli.RecordEpisodeConfig(robot=object(), teleop=object(), task="t")  # type: ignore[arg-type]
    cameras = cli.preset_cameras(config)
    assert list(cameras) == ["wrist"]
    wrist = cameras["wrist"]
    assert isinstance(wrist, OpenCVCameraConfig)
    assert (wrist.index_or_path, wrist.fps, wrist.width, wrist.height) == (0, 30, 640, 480)

    config.cameras = "wrist,front"
    config.wrist_camera = "/dev/video2"
    config.front_camera = "CP1234"
    cameras = cli.preset_cameras(config)
    assert list(cameras) == ["wrist", "front"]
    assert cameras["wrist"].index_or_path == Path("/dev/video2")
    assert isinstance(cameras["front"], OrbbecCameraConfig)
    assert cameras["front"].serial_number_or_name == "CP1234"

    config.cameras = "side"
    with pytest.raises(ValueError, match="--robot.cameras"):
        cli.preset_cameras(config)


def test_real_rebot_configs_get_the_wrist_camera(tmp_path: Path) -> None:
    """The real follower and leader classes, built but not connected, record wrist only."""
    from lerobot.robots.rebot_b601_follower import RebotB601FollowerRobotConfig
    from lerobot.teleoperators.rebot_102_leader import RebotArm102LeaderTeleopConfig

    seen = {}

    def make_robot(robot_cfg):
        from lerobot.robots import make_robot_from_config

        robot = make_robot_from_config(robot_cfg)
        seen["cameras"] = camera_names_for(robot)
        raise RuntimeError("stop before connecting")

    config = cli.RecordEpisodeConfig(
        robot=RebotB601FollowerRobotConfig(port="/dev/null", id="follower"),
        teleop=RebotArm102LeaderTeleopConfig(port="/dev/null", id="leader"),
        task="t",
        output_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="stop before connecting"):
        cli.run(config, make_robot=make_robot, listen_keys=False)
    assert seen["cameras"] == ("wrist",)

    # --robot.cameras wins over the presets.
    from lerobot.cameras.opencv import OpenCVCameraConfig

    explicit = {"side": OpenCVCameraConfig(index_or_path=1, fps=30, width=640, height=480)}
    config.robot = RebotB601FollowerRobotConfig(port="/dev/null", cameras=explicit)
    with pytest.raises(RuntimeError, match="stop before connecting"):
        cli.run(config, make_robot=make_robot, listen_keys=False)
    assert seen["cameras"] == ("side",)

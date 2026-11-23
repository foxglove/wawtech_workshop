"""``rollout-episode``: the MCAP strategy for ``lerobot-rollout``, with no hardware.

The strategy runs against a minimal fake ``RolloutContext``. LeRobot's ``BaseStrategy``
``setup``/``run``/``teardown`` are replaced by stubs (the real ones need a policy and an
arm), and the stubbed ``run`` calls ``_log_telemetry`` the way ``BaseStrategy.run`` does.
"""

from __future__ import annotations

import fnmatch
import json
import socket
import sys
import threading
import time as wall
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from lerobot.configs import parser
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.processor import create_transition
from lerobot.rollout import RolloutConfig, create_strategy
from lerobot.rollout.configs import RolloutStrategyConfig
from lerobot.rollout.strategies.base import BaseStrategy
from mcap.reader import make_reader

from tests.fake_hardware import HEIGHT, WIDTH, FakeFollower, FakeTime
from tests.test_recorder import _metadata, _scalars
from tests.ws_client import FoxgloveTestClient
from wawtech_workshop.dataset.video import decode_h264_messages
from wawtech_workshop.mcap_schema import EPISODE_METADATA_NAME, RECORDING_METADATA_NAME
from wawtech_workshop.recorder.session import (
    EPISODE_GLOB,
    ROLLOUT_GLOB,
    ROLLOUT_PREFIX,
    RecordingSession,
    episode_filename,
    next_episode_index,
)
from wawtech_workshop.rollout import strategy as strategy_module
from wawtech_workshop.rollout.strategy import (
    McapStrategy,
    McapStrategyConfig,
    check_output_dir,
    check_port_free,
)

FPS = 30
CAMERA = "wrist"
STEP_NS = 1_000_000_000 // FPS


# ---------------------------------------------------------------- session changes


def _log(session: RecordingSession, robot: FakeFollower, ticks: int) -> None:
    for tick in range(ticks):
        observation = robot.get_observation()
        action = {key: value + 1.0 for key, value in observation.items() if key.endswith(".pos")}
        session.log_step(
            1_700_000_000_000_000_000 + tick * STEP_NS,
            state=observation,
            action=action,
            images={name: observation[name] for name in session.camera_names},
        )


def _session(tmp_path: Path, robot: FakeFollower, **overrides: Any) -> RecordingSession:
    fields: dict[str, Any] = dict(
        output_dir=tmp_path / "out",
        fps=FPS,
        robot_type=robot.name,
        state_names=tuple(robot.action_features),
        camera_names=(CAMERA,),
        lerobot_version="0.7.0",
        session_id="test",
    )
    fields.update(overrides)
    return RecordingSession(**fields)


def test_filenames_use_the_prefix_and_default_to_episode() -> None:
    assert episode_filename(3, "abc") == "episode_0003_abc.mcap"
    assert episode_filename(3, "abc", "rollout") == "rollout_0003_abc.mcap"
    assert not fnmatch.fnmatch("rollout_0000_abc.mcap", EPISODE_GLOB)
    assert not fnmatch.fnmatch("episode_0000_abc.mcap", ROLLOUT_GLOB)


def test_rollout_index_continues_only_from_rollout_files(tmp_path: Path) -> None:
    (tmp_path / episode_filename(7, "a")).write_bytes(b"")
    assert next_episode_index(tmp_path, ROLLOUT_GLOB) == 0
    (tmp_path / episode_filename(2, "a", ROLLOUT_PREFIX)).write_bytes(b"")
    assert next_episode_index(tmp_path, ROLLOUT_GLOB) == 3
    # Teleop numbering is unchanged and ignores the rollout files.
    assert next_episode_index(tmp_path) == 8


def test_session_writes_prefixed_files_and_rejects_a_bad_prefix(tmp_path: Path) -> None:
    robot = FakeFollower(FakeTime(), cameras=(CAMERA,))
    with _session(tmp_path, robot, file_prefix="rollout") as session:
        path = session.start_episode(task="t", episode_index=4)
        session.discard_episode()
    assert path.name == "rollout_0004_test.mcap"
    assert not path.exists()
    with pytest.raises(ValueError, match="file_prefix"):
        _session(tmp_path, robot, file_prefix="a_b")


def test_unlabeled_episode_omits_outcome_and_merges_extra_metadata(tmp_path: Path) -> None:
    robot = FakeFollower(FakeTime(), cameras=(CAMERA,))
    with _session(tmp_path, robot, file_prefix="rollout") as session:
        session.start_episode(task="t", episode_index=0)
        _log(session, robot, 4)
        assert session.episode_steps == 4
        result = session.finish_episode(None, {"source": "policy", "end_reason": "completed"})
        assert session.episode_steps == 0
    assert result.outcome is None and result.steps == 4
    recording = _metadata(result.path)[RECORDING_METADATA_NAME]
    assert "outcome" not in recording
    assert recording["source"] == "policy" and recording["end_reason"] == "completed"
    assert recording["steps"] == "4" and recording["session_id"] == "test"


def test_labeled_episode_still_writes_the_outcome(tmp_path: Path) -> None:
    robot = FakeFollower(FakeTime(), cameras=(CAMERA,))
    with _session(tmp_path, robot) as session:
        session.start_episode(task="t", episode_index=0)
        _log(session, robot, 2)
        result = session.finish_episode("failure")
    assert result.path.name == "episode_0000_test.mcap"
    assert _metadata(result.path)[RECORDING_METADATA_NAME]["outcome"] == "failure"


@pytest.mark.parametrize("extra", [{"steps": "9"}, {"outcome": "success"}, {"n": 1}])
def test_extra_metadata_cannot_override_session_keys_or_hold_non_strings(
    tmp_path: Path, extra: dict[str, Any]
) -> None:
    robot = FakeFollower(FakeTime(), cameras=(CAMERA,))
    with _session(tmp_path, robot) as session:
        session.start_episode(task="t", episode_index=0)
        _log(session, robot, 2)
        with pytest.raises(ValueError, match="extra_metadata"):
            session.finish_episode(None, extra)
        # The episode is still open, so the caller can fix the call or discard.
        assert session.recording
        session.discard_episode()


# ---------------------------------------------------------------- strategy harness


@dataclass
class Driver:
    """What the stubbed ``BaseStrategy`` methods do, and what they saw."""

    ticks: int = 5
    fail_with: BaseException | None = None
    setup_error: BaseException | None = None
    # (obs_processed, action) of every tick that carried an action.
    logged: list[tuple[dict, dict]] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    follower: FakeFollower | None = None


@dataclass
class Harness:
    strategy: McapStrategy
    ctx: Any
    driver: Driver
    output_dir: Path

    def files(self) -> list[Path]:
        return sorted(self.output_dir.glob("*.mcap")) if self.output_dir.exists() else []


def _fake_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    now = iter(range(1_700_000_000_000_000_000, 2_000_000_000_000_000_000, STEP_NS))
    monkeypatch.setattr(strategy_module.time, "time_ns", lambda: next(now))


def _context(follower: FakeFollower, **runtime: Any) -> Any:
    cfg = dict(
        display_data=False,
        task="pass the butter",
        fps=float(FPS),
        device="cpu",
        policy=SimpleNamespace(pretrained_path="outputs/train/act/pretrained_model", type="act"),
    )
    cfg.update(runtime)
    return SimpleNamespace(
        runtime=SimpleNamespace(cfg=SimpleNamespace(**cfg), shutdown_event=threading.Event()),
        hardware=SimpleNamespace(robot_wrapper=SimpleNamespace(inner=follower)),
        # Deliberately not the observation order: the file's labels must follow this.
        data=SimpleNamespace(ordered_action_keys=list(reversed(list(follower.action_features)))),
    )


def _make(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    stream: bool = False,
    runtime: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    **driver_fields: Any,
) -> Harness:
    _fake_clock(monkeypatch)
    time = FakeTime()
    follower = FakeFollower(time, cameras=(CAMERA,))
    driver = Driver(follower=follower, **driver_fields)

    def fake_setup(self: BaseStrategy, ctx: Any) -> None:
        driver.calls.append("setup")
        if driver.setup_error is not None:
            raise driver.setup_error

    def fake_run(self: BaseStrategy, ctx: Any) -> None:
        driver.calls.append("run")
        for tick in range(driver.ticks):
            observation = follower.get_observation()
            # Warm-up or waiting ticks have no step; held ticks have a step with no action.
            self._log_telemetry(observation, None, ctx.runtime)
            self._log_telemetry(observation, create_transition(), ctx.runtime)
            action = {
                key: float(value) + 0.5
                for key, value in observation.items()
                if key.endswith(".pos")
            }
            self._log_telemetry(observation, create_transition(action=action), ctx.runtime)
            driver.logged.append((observation, action))
        if driver.fail_with is not None:
            raise driver.fail_with

    def fake_teardown(self: BaseStrategy, ctx: Any) -> None:
        driver.calls.append("teardown")

    monkeypatch.setattr(BaseStrategy, "setup", fake_setup)
    monkeypatch.setattr(BaseStrategy, "run", fake_run)
    monkeypatch.setattr(BaseStrategy, "teardown", fake_teardown)

    output_dir = tmp_path / "rollouts"
    strategy_config = McapStrategyConfig(
        output_dir=output_dir,
        stream=stream,
        stream_port=0,
        session_id="test",
        **(config or {}),
    )
    return Harness(
        strategy=create_strategy(strategy_config),
        ctx=_context(follower, **(runtime or {})),
        driver=driver,
        output_dir=output_dir,
    )


def _topics(path: Path) -> dict[str, str]:
    with path.open("rb") as handle:
        summary = make_reader(handle).get_summary()
    assert summary is not None
    return {
        channel.topic: summary.schemas[channel.schema_id].name
        for channel in summary.channels.values()
    }


# ---------------------------------------------------------------- strategy behaviour


def test_one_run_writes_one_rollout_file_with_the_teleop_topics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h = _make(monkeypatch, tmp_path, ticks=6)
    h.strategy.setup(h.ctx)
    h.strategy.run(h.ctx)
    h.strategy.teardown(h.ctx)

    [path] = h.files()
    assert path.name == "rollout_0000_test.mcap"
    assert h.driver.calls == ["setup", "run", "teardown"]
    assert _topics(path) == {
        "/observation/state": "lerobot.Scalars",
        "/action/state": "lerobot.Scalars",
        f"/observation/images/{CAMERA}": "foxglove.CompressedVideo",
        "/joint_states": "foxglove.JointStates",
        "/action/joint_states": "foxglove.JointStates",
        "/tf": "foxglove.FrameTransforms",
    }

    # Labels follow ordered_action_keys, not the observation's own order.
    labels = list(reversed(list(h.driver.follower.action_features)))
    times, state_labels, states = _scalars(path, "/observation/state")
    action_times, action_labels, actions = _scalars(path, "/action/state")
    assert state_labels == action_labels == labels
    assert times == action_times and len(times) == 6
    assert times == sorted(set(times))
    np.testing.assert_allclose(
        states, [[obs[label] for label in labels] for obs, _ in h.driver.logged], rtol=1e-6
    )
    np.testing.assert_allclose(
        actions, [[action[label] for label in labels] for _, action in h.driver.logged], rtol=1e-6
    )

    with path.open("rb") as handle:
        messages = [
            (message.log_time, json.loads(message.data))
            for _s, _c, message in make_reader(handle).iter_messages(
                topics=[f"/observation/images/{CAMERA}"]
            )
        ]
    frames = decode_h264_messages(messages)
    assert [stamp for stamp, _ in frames] == times
    assert frames[0][1].shape == (HEIGHT, WIDTH, 3)

    metadata = _metadata(path)
    episode = metadata[EPISODE_METADATA_NAME]
    assert episode["task"] == "pass the butter"
    assert episode["fps"] == str(FPS) and episode["episode_index"] == "0"
    assert episode["robot_type"] == "rebot_b601_follower"
    recording = metadata[RECORDING_METADATA_NAME]
    assert "outcome" not in recording
    assert recording["source"] == "policy"
    assert recording["policy_path"] == "outputs/train/act/pretrained_model"
    assert recording["policy_type"] == "act"
    assert recording["device"] == "cpu"
    assert recording["end_reason"] == "completed"
    assert recording["steps"] == "6" and recording["session_id"] == "test"


def test_a_run_with_no_logged_steps_leaves_no_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    h = _make(monkeypatch, tmp_path, ticks=0)
    h.strategy.setup(h.ctx)
    h.strategy.run(h.ctx)
    h.strategy.teardown(h.ctx)
    assert h.files() == []
    assert "No steps were logged" in capsys.readouterr().out


def test_each_run_is_its_own_file_and_the_session_survives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Interactive mode calls run() once per /start segment."""
    h = _make(monkeypatch, tmp_path, ticks=3)
    h.strategy.setup(h.ctx)
    h.strategy.run(h.ctx)
    session = h.strategy._session
    assert session is not None and not session.recording
    h.strategy.run(h.ctx)
    assert h.strategy._session is session
    h.strategy.teardown(h.ctx)
    assert [path.name for path in h.files()] == ["rollout_0000_test.mcap", "rollout_0001_test.mcap"]
    assert h.driver.calls == ["setup", "run", "run", "teardown"]
    for path in h.files():
        assert _metadata(path)[RECORDING_METADATA_NAME]["steps"] == "3"


def test_indices_continue_from_existing_rollouts_but_not_from_episodes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h = _make(monkeypatch, tmp_path, ticks=2)
    h.output_dir.mkdir()
    (h.output_dir / "rollout_0004_old.mcap").write_bytes(b"")
    h.strategy.setup(h.ctx)
    h.strategy.run(h.ctx)
    h.strategy.teardown(h.ctx)
    assert "rollout_0005_test.mcap" in [path.name for path in h.files()]


def test_a_failing_run_still_saves_its_file_marked_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h = _make(monkeypatch, tmp_path, ticks=4, fail_with=RuntimeError("policy fault"))
    h.strategy.setup(h.ctx)
    with pytest.raises(RuntimeError, match="policy fault"):
        h.strategy.run(h.ctx)
    [path] = h.files()
    recording = _metadata(path)[RECORDING_METADATA_NAME]
    assert recording["end_reason"] == "error" and recording["steps"] == "4"
    assert "outcome" not in recording
    # The strategy is still usable: the next segment records a new file.
    h.driver.fail_with = None
    h.strategy.run(h.ctx)
    assert len(h.files()) == 2
    h.strategy.teardown(h.ctx)


def test_a_run_that_hits_its_duration_is_completed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h = _make(monkeypatch, tmp_path, ticks=3)
    assert not h.ctx.runtime.shutdown_event.is_set()
    h.strategy.setup(h.ctx)
    h.strategy.run(h.ctx)
    h.strategy.teardown(h.ctx)
    [path] = h.files()
    assert _metadata(path)[RECORDING_METADATA_NAME]["end_reason"] == "completed"


def test_a_run_ended_by_the_shutdown_event_is_stopped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ctrl-C and SIGTERM only set the event; BaseStrategy.run then returns normally."""
    h = _make(monkeypatch, tmp_path, ticks=3)
    h.strategy.setup(h.ctx)
    h.ctx.runtime.shutdown_event.set()
    h.strategy.run(h.ctx)
    h.strategy.teardown(h.ctx)
    [path] = h.files()
    recording = _metadata(path)[RECORDING_METADATA_NAME]
    assert recording["end_reason"] == "stopped" and recording["steps"] == "3"


def test_the_server_is_announced_as_a_rollout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}
    real = strategy_module.start_stream_server

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(strategy_module, "start_stream_server", spy)
    h = _make(monkeypatch, tmp_path, stream=True)
    h.strategy.setup(h.ctx)
    h.strategy.teardown(h.ctx)
    assert seen["name"] == "wawtech rollout"


def test_ctrl_c_is_marked_interrupted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    h = _make(monkeypatch, tmp_path, ticks=2, fail_with=KeyboardInterrupt())
    h.strategy.setup(h.ctx)
    with pytest.raises(KeyboardInterrupt):
        h.strategy.run(h.ctx)
    h.strategy.teardown(h.ctx)
    [path] = h.files()
    assert _metadata(path)[RECORDING_METADATA_NAME]["end_reason"] == "interrupted"


def test_a_failing_run_with_no_steps_leaves_no_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h = _make(monkeypatch, tmp_path, ticks=0, fail_with=RuntimeError("boom"))
    h.strategy.setup(h.ctx)
    with pytest.raises(RuntimeError, match="boom"):
        h.strategy.run(h.ctx)
    h.strategy.teardown(h.ctx)
    assert h.files() == []


def test_a_save_failure_does_not_replace_the_run_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h = _make(monkeypatch, tmp_path, ticks=2, fail_with=RuntimeError("policy fault"))
    h.strategy.setup(h.ctx)

    def broken(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(h.strategy._session, "finish_episode", broken)
    with pytest.raises(RuntimeError, match="policy fault"):
        h.strategy.run(h.ctx)
    assert not h.strategy._session.recording
    h.strategy.teardown(h.ctx)
    assert h.files() == []


def test_a_save_failure_after_a_clean_run_is_raised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h = _make(monkeypatch, tmp_path, ticks=2)
    h.strategy.setup(h.ctx)

    def broken(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(h.strategy._session, "finish_episode", broken)
    with pytest.raises(OSError, match="disk full"):
        h.strategy.run(h.ctx)
    assert not h.strategy._session.recording
    h.strategy.teardown(h.ctx)


def test_only_policy_action_ticks_are_logged_when_interpolating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h = _make(monkeypatch, tmp_path, ticks=0)
    h.strategy.setup(h.ctx)
    h.strategy._interpolator = SimpleNamespace(emitted_policy_action=False)
    h.strategy.run(h.ctx)
    assert h.files() == []
    h.strategy._interpolator.emitted_policy_action = True
    h.driver.ticks = 2
    h.strategy.run(h.ctx)
    [path] = h.files()
    assert _metadata(path)[RECORDING_METADATA_NAME]["steps"] == "2"
    h.strategy.teardown(h.ctx)


def test_teardown_always_reaches_the_parent_even_if_closing_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h = _make(monkeypatch, tmp_path, ticks=1)
    h.strategy.setup(h.ctx)
    session = h.strategy._session

    def broken_close() -> None:
        raise OSError("close failed")

    monkeypatch.setattr(session, "close", broken_close)
    with pytest.raises(OSError, match="close failed"):
        h.strategy.teardown(h.ctx)
    assert h.driver.calls[-1] == "teardown"
    assert h.strategy._session is None
    # A second teardown (nothing left to close) is harmless.
    h.strategy.teardown(h.ctx)
    assert h.driver.calls.count("teardown") == 2
    session.close = type(session).close.__get__(session)
    session.close()


def test_teardown_reaches_the_parent_if_the_server_fails_to_stop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h = _make(monkeypatch, tmp_path, ticks=1, stream=True)
    h.strategy.setup(h.ctx)
    real_stop = h.strategy._server.stop

    class BrokenServer:
        port = h.strategy._server.port

        def stop(self) -> None:
            real_stop()
            raise OSError("stop failed")

    h.strategy._server = BrokenServer()
    with pytest.raises(OSError, match="stop failed"):
        h.strategy.teardown(h.ctx)
    assert h.driver.calls[-1] == "teardown"
    assert h.strategy._session is None


def test_teardown_without_a_session_still_calls_the_parent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """LeRobot calls teardown from ``finally`` even when setup raised."""
    h = _make(monkeypatch, tmp_path)
    h.strategy.teardown(h.ctx)
    assert h.driver.calls == ["teardown"]


def test_display_data_is_rejected_before_anything_is_created(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h = _make(monkeypatch, tmp_path, runtime={"display_data": True})
    with pytest.raises(ValueError, match="display_data"):
        h.strategy.setup(h.ctx)
    assert h.strategy._session is None and h.strategy._server is None
    assert h.driver.calls == []
    assert not h.output_dir.exists()


@pytest.mark.parametrize(
    ("runtime", "message"),
    [
        ({"task": "  "}, "--task is required"),
        ({"fps": 29.97}, "whole number"),
    ],
)
def test_setup_rejects_what_the_recording_cannot_use(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runtime: dict[str, Any], message: str
) -> None:
    h = _make(monkeypatch, tmp_path, runtime=runtime)
    with pytest.raises(ValueError, match=message):
        h.strategy.setup(h.ctx)
    assert h.strategy._session is None and h.driver.calls == []


def test_setup_needs_an_rgb_camera_and_every_joint_observed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h = _make(monkeypatch, tmp_path)
    h.ctx.hardware.robot_wrapper.inner.cameras = ()
    with pytest.raises(ValueError, match="no RGB cameras"):
        h.strategy.setup(h.ctx)
    h2 = _make(monkeypatch, tmp_path)
    h2.ctx.data.ordered_action_keys = [*h2.ctx.data.ordered_action_keys, "wrist_roll.pos"]
    with pytest.raises(ValueError, match="wrist_roll.pos"):
        h2.strategy.setup(h2.ctx)


def test_a_setup_failure_after_the_session_exists_cleans_it_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h = _make(monkeypatch, tmp_path, stream=True, setup_error=RuntimeError("engine failed"))
    with pytest.raises(RuntimeError, match="engine failed"):
        h.strategy.setup(h.ctx)
    assert h.strategy._session is None and h.strategy._server is None
    # LeRobot still calls teardown, which must disconnect the arm.
    h.strategy.teardown(h.ctx)
    assert h.driver.calls == ["setup", "teardown"]


def test_a_taken_stream_port_at_setup_cleans_up_and_names_the_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import foxglove

    h = _make(monkeypatch, tmp_path, stream=True)
    taken = foxglove.start_server(name="other", port=0, context=foxglove.Context())
    try:
        h.strategy.config.stream_port = taken.port
        with pytest.raises(RuntimeError, match="--strategy.stream=false"):
            h.strategy.setup(h.ctx)
    finally:
        taken.stop()
    assert h.strategy._session is None and h.strategy._server is None
    assert h.driver.calls == []


def test_the_rollout_streams_live_while_writing_the_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    topics = [
        "/observation/state",
        "/action/state",
        f"/observation/images/{CAMERA}",
        "/joint_states",
    ]
    h = _make(monkeypatch, tmp_path, stream=True, ticks=5)
    h.strategy.setup(h.ctx)
    client = FoxgloveTestClient(h.strategy._server.port)
    try:
        client.subscribe(topics)
        wall.sleep(0.3)  # let the server register the subscription before the run
        h.strategy.run(h.ctx)
        h.strategy.teardown(h.ctx)
        client.closed.wait(timeout=5)
    finally:
        client.close()
    assert "Streaming live: open ws://127.0.0.1:" in capsys.readouterr().out
    for topic in topics:
        assert client.messages[topic], f"nothing streamed on {topic}"
    [path] = h.files()
    assert len(_scalars(path, "/observation/state")[0]) == 5


def test_stream_off_starts_no_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def no_server(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("--strategy.stream=false must not start a server")

    monkeypatch.setattr(strategy_module, "start_stream_server", no_server)
    h = _make(monkeypatch, tmp_path, stream=False)
    h.strategy.setup(h.ctx)
    assert h.strategy._server is None
    h.strategy.teardown(h.ctx)


# ---------------------------------------------------------------- config guards


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "data" / "episodes").mkdir(parents=True)
    return tmp_path


@pytest.mark.parametrize(
    "folder",
    [
        "data/episodes",
        "data",
        "data/episodes/session2",  # does not exist yet
        "./data/../data/episodes",
        "data/other",
    ],
)
def test_output_dir_inside_data_is_refused(repo: Path, folder: str) -> None:
    with pytest.raises(ValueError, match="training episodes.*Foxlet"):
        check_output_dir(Path(folder), cwd=repo)
    with pytest.raises(ValueError, match="Foxlet"):
        check_output_dir(repo / folder, cwd=repo)  # absolute path


def test_output_dir_reaching_data_through_a_symlink_is_refused(repo: Path) -> None:
    (repo / "outputs").mkdir()
    (repo / "outputs" / "link").symlink_to(repo / "data" / "episodes", target_is_directory=True)
    with pytest.raises(ValueError, match="Foxlet"):
        check_output_dir(Path("outputs/link"), cwd=repo)


def test_a_folder_holding_teleop_episodes_is_refused(repo: Path) -> None:
    folder = repo / "mine"
    folder.mkdir()
    (folder / episode_filename(0, "x")).write_bytes(b"")
    with pytest.raises(ValueError, match="already holds teleop episodes"):
        check_output_dir(Path("mine"), cwd=repo)


@pytest.mark.parametrize("folder", ["outputs/rollouts", "data2", "database", "elsewhere/data"])
def test_other_folders_are_accepted(repo: Path, folder: str) -> None:
    check_output_dir(Path(folder), cwd=repo)
    (repo / folder).mkdir(parents=True, exist_ok=True)
    (repo / folder / "rollout_0000_x.mcap").write_bytes(b"")  # rollouts alone are fine
    check_output_dir(Path(folder), cwd=repo)


def test_config_checks_run_at_construction(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(repo)
    with pytest.raises(ValueError, match="Foxlet"):
        McapStrategyConfig(output_dir=Path("data/episodes"), stream=False)
    config = McapStrategyConfig(output_dir=Path("outputs/rollouts"), stream_port=0)
    assert config.output_dir == Path("outputs/rollouts")
    assert config.type == "mcap" and config.stream
    assert (config.stream_host, config.stream_port) == ("127.0.0.1", 0)
    assert McapStrategyConfig.supports_interactive is True
    assert McapStrategyConfig.dataset_mode == "none"


def test_a_stream_port_in_use_fails_at_construction_not_at_connect(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(repo)
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        with pytest.raises(RuntimeError, match=rf"127.0.0.1:{port}.*already in use.*--strategy"):
            McapStrategyConfig(stream_port=port)
        with pytest.raises(RuntimeError, match="--strategy.stream=false"):
            check_port_free("127.0.0.1", port)
        # Not streaming, or letting the OS pick the port, needs no free port.
        McapStrategyConfig(stream=False, stream_port=port)
        check_port_free("127.0.0.1", 0)
    check_port_free("127.0.0.1", port)  # free again once released


def test_an_unresolvable_stream_host_is_reported(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(repo)
    with pytest.raises(RuntimeError, match="Cannot stream on"):
        McapStrategyConfig(stream_host="no-such-host.invalid", stream_port=8765)


# ---------------------------------------------------------------- LeRobot wiring


def test_mcap_is_a_known_strategy_choice_and_builds_the_strategy(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(repo)
    assert RolloutStrategyConfig.get_known_choices()["mcap"] is McapStrategyConfig
    strategy = create_strategy(McapStrategyConfig(stream=False))
    assert type(strategy) is McapStrategy


def _parse_rollout(monkeypatch: pytest.MonkeyPatch, *args: str) -> RolloutConfig:
    def parse(cfg):  # noqa: ANN001, ANN202
        return cfg

    # LeRobot's parser reads the real annotation, which `from __future__ import annotations`
    # would turn into a string.
    parse.__annotations__ = {"cfg": RolloutConfig}
    monkeypatch.setattr(sys, "argv", ["rollout-episode", *args])
    return parser.wrap()(parse)()


@pytest.fixture
def policy_dir(tmp_path: Path) -> Path:
    path = tmp_path / "policy"
    ACTConfig().save_pretrained(path)
    return path


def test_a_full_rollout_config_parses_with_the_mcap_strategy(
    repo: Path, policy_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(repo)
    cfg = _parse_rollout(
        monkeypatch,
        "--strategy.type=mcap",
        "--strategy.stream_port=0",
        "--strategy.output_dir=outputs/rollouts",
        "--strategy.session_id=demo",
        "--robot.type=rebot_b601_follower",
        "--robot.port=/dev/null",
        f"--policy.path={policy_dir}",
        "--policy.device=cpu",
        "--task=pass the butter",
        "--duration=20",
        "--interactive=true",
    )
    assert isinstance(cfg.strategy, McapStrategyConfig)
    assert cfg.strategy.session_id == "demo" and cfg.strategy.output_dir == Path("outputs/rollouts")
    assert cfg.policy.type == "act" and str(cfg.policy.pretrained_path) == str(policy_dir)
    assert cfg.interactive and cfg.display_data is False
    assert type(create_strategy(cfg.strategy)) is McapStrategy


def _chain(exc: BaseException | None) -> str:
    texts = []
    while exc is not None:
        texts.append(str(exc))
        exc = exc.__cause__
    return "\n".join(texts)


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--strategy.output_dir=data/episodes"], "Foxlet"),
        (["--dataset.repo_id=me/rollout_x"], "does not record data"),
    ],
)
def test_a_bad_rollout_config_is_refused_while_parsing(
    repo: Path,
    policy_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    flags: list[str],
    message: str,
) -> None:
    """draccus wraps the ValueError, so the message is found along the exception chain."""
    monkeypatch.chdir(repo)
    with pytest.raises(Exception) as caught:  # noqa: B017
        _parse_rollout(
            monkeypatch,
            "--strategy.type=mcap",
            "--strategy.stream_port=0",
            "--robot.type=rebot_b601_follower",
            "--robot.port=/dev/null",
            f"--policy.path={policy_dir}",
            "--policy.device=cpu",
            *flags,
        )
    assert message in _chain(caught.value)

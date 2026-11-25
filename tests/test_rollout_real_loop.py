"""The real ``BaseStrategy.run`` loop driving ``McapStrategy``, offline.

Real here: LeRobot's config parser, ``build_rollout_context`` (policy loading, processors,
inference engine, feature wiring, hardware connect), ``create_strategy``, the real
``BaseStrategy.run`` with its ``CycleTimer``, ``send_next_action`` and
``ActionInterpolator``, our ``_log_telemetry`` override, the real wall clock and the real
H.264 encoder.

Stubbed: the robot (``tests/fake_lerobot_robot.py``, registered with LeRobot as
``--robot.type=wawtech_fake``) and the policy weights, a tiny randomly initialised ACT with
its normaliser processors saved to a tmp dir. The ``rollout()`` entry point's signal handler
and ``InteractiveSession`` are not exercised: the test calls the same
``build_rollout_context`` / ``setup`` / ``run`` / ``teardown`` sequence ``rollout()`` does.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest
import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.rollout import build_rollout_context, create_strategy
from mcap.reader import make_reader

from tests.fake_lerobot_robot import CAMERA, HEIGHT, JOINTS, WIDTH, WawtechFakeRobot
from tests.test_recorder import _metadata, _scalars
from tests.test_rollout import _parse_rollout, _topics
from wawtech_workshop.compressed_video import parse_compressed_video
from wawtech_workshop.dataset.video import decode_h264_messages
from wawtech_workshop.mcap_schema import EPISODE_METADATA_NAME, RECORDING_METADATA_NAME
from wawtech_workshop.rollout.strategy import McapStrategy

FPS = 30
DURATION_S = 1.5
EXPECTED_STEPS = int(FPS * DURATION_S)
LABELS = [f"{joint}.pos" for joint in JOINTS]


@pytest.fixture(scope="module")
def policy_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny random ACT policy with its processors, as ``lerobot-train`` saves them."""
    path = tmp_path_factory.mktemp("policy")
    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            f"observation.images.{CAMERA}": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, HEIGHT, WIDTH)
            ),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        chunk_size=5,
        n_action_steps=5,
        pretrained_backbone_weights=None,  # nothing to download
        device="cpu",
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
    )
    torch.manual_seed(0)
    ACTPolicy(config).save_pretrained(path)
    stats = {
        "observation.state": {"mean": torch.zeros(3), "std": torch.full((3,), 10.0)},
        f"observation.images.{CAMERA}": {
            "mean": torch.zeros(3, 1, 1),
            "std": torch.ones(3, 1, 1),
        },
        "action": {"mean": torch.zeros(3), "std": torch.full((3,), 10.0)},
    }
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    preprocessor.save_pretrained(path)
    postprocessor.save_pretrained(path)
    return path


def _rollout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    policy_dir: Path,
    *flags: str,
    shutdown: threading.Event | None = None,
) -> tuple[Path, WawtechFakeRobot]:
    """What ``lerobot_rollout.rollout`` does after parsing, minus signals and the prompt."""
    monkeypatch.chdir(tmp_path)
    WawtechFakeRobot.built.clear()
    output_dir = tmp_path / "rollouts"
    cfg = _parse_rollout(
        monkeypatch,
        "--strategy.type=mcap",
        "--strategy.stream_port=0",
        f"--strategy.output_dir={output_dir}",
        "--strategy.session_id=real",
        "--robot.type=wawtech_fake",
        f"--policy.path={policy_dir}",
        "--policy.device=cpu",
        "--device=cpu",
        "--task=pass the butter",
        f"--fps={FPS}",
        f"--duration={DURATION_S}",
        "--return_to_initial_position=false",
        *flags,
    )
    ctx = build_rollout_context(cfg, shutdown or threading.Event())
    strategy = create_strategy(cfg.strategy)
    assert type(strategy) is McapStrategy
    try:
        strategy.setup(ctx)
        strategy.run(ctx)
    finally:
        strategy.teardown(ctx)
    [robot] = WawtechFakeRobot.built
    assert not robot.is_connected, "teardown must disconnect the arm"
    return output_dir, robot


def _check_file(
    output_dir: Path, robot: WawtechFakeRobot, policy_dir: Path, end_reason: str
) -> tuple[Path, int]:
    [path] = sorted(output_dir.glob("*.mcap"))
    assert path.name == "rollout_0000_real.mcap"
    assert _topics(path) == {
        "/observation/state": "lerobot.Scalars",
        "/action/state": "lerobot.Scalars",
        f"/observation/images/{CAMERA}": "foxglove.CompressedVideo",
        "/joint_states": "foxglove.JointStates",
        "/action/joint_states": "foxglove.JointStates",
        "/tf": "foxglove.FrameTransforms",
    }
    times, state_labels, states = _scalars(path, "/observation/state")
    action_times, action_labels, actions = _scalars(path, "/action/state")
    assert state_labels == action_labels == LABELS
    assert times == action_times and times == sorted(set(times))
    steps = len(times)

    with path.open("rb") as handle:
        messages = [
            (message.log_time, parse_compressed_video(message.data))
            for _s, _c, message in make_reader(handle).iter_messages(
                topics=[f"/observation/images/{CAMERA}"]
            )
        ]
    frames = decode_h264_messages(messages)
    assert [stamp for stamp, _ in frames] == times
    assert frames[0][1].shape == (HEIGHT, WIDTH, 3)

    metadata = _metadata(path)
    assert metadata[EPISODE_METADATA_NAME]["fps"] == str(FPS)
    assert metadata[EPISODE_METADATA_NAME]["task"] == "pass the butter"
    assert metadata[EPISODE_METADATA_NAME]["robot_type"] == "wawtech_fake"
    recording = metadata[RECORDING_METADATA_NAME]
    assert "outcome" not in recording
    assert recording["source"] == "policy"
    assert recording["policy_type"] == "act" and recording["device"] == "cpu"
    assert recording["policy_path"] == str(policy_dir)
    assert recording["end_reason"] == end_reason
    assert recording["steps"] == str(steps)

    # The state is the robot's observation: its joints sit where the last command put them.
    sent = np.asarray([[row[label] for label in LABELS] for row in robot.sent])
    assert len(sent) > 0 and np.isfinite(sent).all() and np.abs(sent).max() > 0
    assert np.isfinite(states).all() and np.isfinite(actions).all()
    # Every recorded action is one the robot was sent, in order.
    positions = [
        next(i for i, row in enumerate(sent) if np.allclose(row, action, rtol=1e-5, atol=1e-6))
        for action in actions
    ]
    assert positions == sorted(positions)
    return path, steps


def test_the_real_loop_records_one_file_at_the_rollout_fps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, policy_dir: Path
) -> None:
    output_dir, robot = _rollout(monkeypatch, tmp_path, policy_dir)
    path, steps = _check_file(output_dir, robot, policy_dir, "completed")
    assert 0.6 * EXPECTED_STEPS <= steps <= EXPECTED_STEPS + 5, steps
    # With no interpolation, one command is sent per recorded step.
    assert len(robot.sent) == steps
    recorded = _scalars(path, "/action/state")[2]
    np.testing.assert_allclose(
        recorded, [[row[label] for label in LABELS] for row in robot.sent], rtol=1e-5, atol=1e-6
    )


def test_interpolating_commands_do_not_double_the_recorded_rate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, policy_dir: Path
) -> None:
    output_dir, robot = _rollout(monkeypatch, tmp_path, policy_dir, "--interpolation_multiplier=2")
    _path, steps = _check_file(output_dir, robot, policy_dir, "completed")
    assert 0.6 * EXPECTED_STEPS <= steps <= EXPECTED_STEPS + 5, steps
    # The robot gets commands at twice the rate; the file only records the policy's own.
    assert len(robot.sent) >= int(1.6 * steps), (len(robot.sent), steps)


def test_a_shutdown_requested_mid_run_is_recorded_as_stopped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, policy_dir: Path
) -> None:
    """The signal handler only sets the event: the real loop returns, we say ``stopped``."""
    shutdown = threading.Event()
    real_get_observation = WawtechFakeRobot.get_observation

    def get_observation(self: WawtechFakeRobot):
        observation = real_get_observation(self)
        if self.observations >= 21:  # the first read is at connect, the rest are ticks
            shutdown.set()
        return observation

    monkeypatch.setattr(WawtechFakeRobot, "get_observation", get_observation)
    output_dir, robot = _rollout(
        monkeypatch, tmp_path, policy_dir, "--duration=30", shutdown=shutdown
    )
    _path, steps = _check_file(output_dir, robot, policy_dir, "stopped")
    assert 10 <= steps <= 21, steps

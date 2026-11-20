"""The arm model for Foxglove's 3D panel: /joint_states, /tf and the served URDF."""

from __future__ import annotations

import math
import struct
import time as wall
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from mcap.reader import make_reader

from tests.fake_hardware import FakeFollower, FakeLeader, FakeTime
from tests.test_recorder import _record, _streaming_config, _with_fake_clock
from tests.ws_client import FoxgloveTestClient
from wawtech_workshop.recorder import cli, urdf


def _fields(data: bytes) -> list[tuple[int, int | bytes]]:
    """Top-level (field number, value) pairs of one protobuf message."""
    fields, i = [], 0

    def varint() -> int:
        nonlocal i
        value = shift = 0
        while True:
            byte = data[i]
            i += 1
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return value
            shift += 7

    while i < len(data):
        key = varint()
        number, wire = key >> 3, key & 7
        if wire == 0:
            fields.append((number, varint()))
        elif wire == 1:
            fields.append((number, data[i : i + 8]))
            i += 8
        elif wire == 2:
            length = varint()
            fields.append((number, data[i : i + length]))
            i += length
        else:
            raise ValueError(f"wire type {wire}")
    return fields


def _joint_states(data: bytes) -> dict[str, float]:
    """name -> position of one ``foxglove.JointStates`` message (joints are field 2)."""
    joints = {}
    for number, value in _fields(data):
        if number == 2:
            joint = dict(_fields(value))
            joints[joint[1].decode()] = struct.unpack("<d", joint[2])[0]
    return joints


def _frame_ids(data: bytes) -> tuple[str, str]:
    """(parent, child) of the first FrameTransform in a ``FrameTransforms`` (field 1)."""
    transform = dict(_fields(dict(_fields(data))[1]))
    return transform[2].decode(), transform[3].decode()


REBOT_STATE = tuple(f"{joint}.pos" for joint in urdf.URDF_JOINTS) + ("gripper.pos",)


def test_joint_names_follow_the_urdf_and_skip_the_gripper() -> None:
    joints = urdf.urdf_joint_names(REBOT_STATE)
    assert list(joints.values()) == ["joint1", "joint2", "join3", "joint4", "joint5", "joint6"]
    assert "gripper.pos" not in joints
    state = {label: 10.0 * (i + 1) for i, label in enumerate(REBOT_STATE)}
    assert urdf.joint_positions_rad(joints, state)[2] == ("join3", pytest.approx(math.radians(30)))
    assert urdf.urdf_joint_names(("something.pos",)) == {}


def test_mapping_matches_the_bundled_urdf() -> None:
    tree = ET.parse(urdf.ASSET_ROOT / urdf.PACKAGE / "urdf" / "reBot-DevArm_fixend.urdf")
    revolute = [j.get("name") for j in tree.findall("joint") if j.get("type") == "revolute"]
    assert revolute == list(urdf.URDF_JOINTS.values())
    # Every mesh the URDF names is in the package, and there is no OAK mount left.
    # Meshes are relative to the URDF, so the app finds them next to it however it loads
    # the file (from disk, or from the recorder as package://<name>/urdf/...).
    urdf_dir = urdf.ASSET_ROOT / urdf.PACKAGE / "urdf"
    for mesh in tree.iter("mesh"):
        filename = mesh.get("filename")
        assert filename.startswith("../meshes/"), filename
        assert (urdf_dir / filename).resolve().is_file(), f"{filename} is missing"
        served = urdf.make_asset_handler()(f"package://{urdf.PACKAGE}/{filename[3:]}")
        assert served, f"{filename} is not served"
    assert not [link for link in tree.findall("link") if "oak" in link.get("name", "")]


def test_asset_handler_serves_only_the_package(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.urdf").write_text("<robot/>")
    (tmp_path.parent / "secret.txt").write_text("nope")
    handler = urdf.make_asset_handler(tmp_path)
    assert handler("package://pkg/a.urdf") == b"<robot/>"
    assert handler("package://pkg/missing.urdf") is None
    assert handler("package://../secret.txt") is None
    assert handler("package://pkg/../../secret.txt") is None
    assert handler("package:///etc/passwd") is None
    assert handler("file:///etc/passwd") is None
    assert handler("https://example.com/a.urdf") is None


def test_episodes_record_joint_states_in_radians(tmp_path: Path) -> None:
    results, _robot, _leader = _record(tmp_path, num_episodes=1, cameras=("wrist",))
    follower: list[tuple[int, dict[str, float]]] = []
    leader: list[tuple[int, dict[str, float]]] = []
    tf: list[tuple[str, str]] = []
    scalar_times: list[int] = []
    with results[0].path.open("rb") as handle:
        reader = make_reader(handle)
        for _schema, channel, message in reader.iter_messages():
            if channel.topic == "/joint_states":
                follower.append((message.log_time, _joint_states(message.data)))
            elif channel.topic == "/action/joint_states":
                leader.append((message.log_time, _joint_states(message.data)))
            elif channel.topic == "/tf":
                tf.append(_frame_ids(message.data))
            elif channel.topic == "/observation/state":
                scalar_times.append(message.log_time)
    assert len(follower) == len(leader) == len(tf) == len(scalar_times) > 0
    assert [t for t, _ in follower] == [t for t, _ in leader] == scalar_times, "one per tick"
    # The fake follower has shoulder_pan, elbow_flex and gripper; the gripper has no URDF joint.
    assert list(follower[0][1]) == list(leader[0][1]) == ["joint1", "join3"]
    assert all(abs(v) <= math.radians(11) for v in follower[0][1].values()), "not radians"
    # The fake leader is the follower's motion plus 1 degree: the ghost is that action.
    for name in ("joint1", "join3"):
        assert leader[0][1][name] - follower[0][1][name] == pytest.approx(math.radians(1))
    assert tf[0] == ("world", "base_link")


@pytest.mark.parametrize("control", ["keys", "foxglove"])
def test_foxglove_fetches_the_urdf_and_meshes_from_the_recorder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control: str
) -> None:
    """The recorder advertises and serves the model in both control modes (the
    foxglove mode passes its own capability list, which must not drop assets)."""
    time = FakeTime()
    robot, leader = FakeFollower(time, cameras=("wrist",)), FakeLeader(time)
    monkeypatch.setattr(cli, "record_episodes", _with_fake_clock(cli.record_episodes, time))
    seen: dict[str, object] = {}
    real = cli.start_stream

    def fetch(cfg, context, controls=None):
        server = real(cfg, context, controls)
        client = FoxgloveTestClient(server.port)
        wall.sleep(0.3)
        seen["capabilities"] = client.capabilities
        seen["urdf"] = client.fetch_asset(urdf.URDF_URI)
        seen["mesh"] = client.fetch_asset(f"package://{urdf.PACKAGE}/meshes/link1.STL")
        seen["traversal"] = client.fetch_asset(f"package://{urdf.PACKAGE}/../../../README.md")
        client.close()
        return server

    monkeypatch.setattr(cli, "start_stream", fetch)
    leader.on_tick = lambda call: None
    if control == "foxglove":
        # Quit as soon as the idle phase starts, so the run ends without an episode.
        from wawtech_workshop.recorder import loop

        original = loop.EpisodeControls.__init__

        def quit_immediately(self, *args, **kwargs):
            original(self, *args, **kwargs)
            self.stop = True

        monkeypatch.setattr(loop.EpisodeControls, "__init__", quit_immediately)
    cli.run(
        _streaming_config(tmp_path, control=control, episode_time_s=0.2),
        make_robot=lambda _cfg: robot,
        make_teleop=lambda _cfg: leader,
        listen_keys=False,
    )
    assert "assets" in seen["capabilities"], seen
    assert seen["urdf"].startswith(b"<?xml") and b"joint1" in seen["urdf"]
    assert seen["mesh"], "meshes are served"
    assert seen["traversal"] is None


def test_missed_and_served_assets_are_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.urdf").write_text("<robot/>")
    handler = urdf.make_asset_handler(tmp_path)
    with caplog.at_level("INFO", logger=urdf.logger.name):
        handler("package://pkg/a.urdf")
        handler("package://pkg/nope.stl")
    assert "Served package://pkg/a.urdf" in caplog.text
    assert "package://pkg/nope.stl: not found" in caplog.text
    caplog.clear()
    assert urdf.check_assets(tmp_path) is False
    assert "Arm model not found" in caplog.text
    assert urdf.check_assets() is True


def test_package_name_is_lowercase_because_the_app_lowercases_it() -> None:
    """Foxglove asks for package://<name lower-cased>/..., so a mixed-case folder is not found."""
    assert urdf.PACKAGE == urdf.PACKAGE.lower()
    handler = urdf.make_asset_handler()
    for name in ("base_link", "link3", "end_link"):
        assert handler(f"package://{urdf.PACKAGE}/meshes/{name}.STL"), name
    assert (urdf.ASSET_ROOT / urdf.PACKAGE).is_dir()
    assert [p.name for p in urdf.ASSET_ROOT.iterdir() if p.is_dir()] == [urdf.PACKAGE]

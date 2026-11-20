"""Starting and finishing episodes from Foxglove: services, status and the idle phase."""

from __future__ import annotations

import json
import time as wall
from pathlib import Path

import pytest
from mcap.reader import make_reader

from tests.fake_hardware import FakeFollower, FakeLeader, FakeTime
from tests.test_recorder import _session, _streaming_config, _with_fake_clock
from tests.ws_client import FoxgloveTestClient
from wawtech_workshop.recorder import cli
from wawtech_workshop.recorder.control import (
    SERVICE_DISCARD,
    SERVICE_FINISH,
    SERVICE_QUIT,
    SERVICE_START,
    TOPIC_STATUS,
    handle_request,
    require_loopback,
)
from wawtech_workshop.recorder.loop import (
    FAILURE,
    IDLE,
    RECORDING,
    SUCCESS,
    EpisodeControls,
    record_episodes,
)


def _call(controls: EpisodeControls, service: str, request: dict | bytes) -> dict:
    payload = request if isinstance(request, bytes) else json.dumps(request).encode()
    return handle_request(controls, service, payload)


def test_requests_are_refused_in_the_wrong_state() -> None:
    controls = EpisodeControls()
    assert controls.phase == IDLE
    for service in (SERVICE_FINISH, SERVICE_DISCARD):
        response = _call(controls, service, {})
        assert response["ok"] is False and "idle" in response["error"]
    assert not controls.episode_should_end()

    assert _call(controls, SERVICE_START, {"task": "  stack the cups "}) == {
        "ok": True,
        "state": IDLE,
    }
    assert controls.start and controls.task == "stack the cups"

    controls.set_phase(RECORDING, 3)
    refused = _call(controls, SERVICE_START, {})
    assert refused["ok"] is False and "recording" in refused["error"]
    assert _call(controls, SERVICE_FINISH, {"outcome": "failure"})["ok"]
    assert controls.take_episode_end() == (FAILURE, False)
    assert _call(controls, SERVICE_FINISH, {})["ok"]  # outcome defaults to success
    assert controls.take_episode_end() == (SUCCESS, False)
    assert _call(controls, SERVICE_DISCARD, b"")["ok"]  # an empty body is {}
    assert controls.take_episode_end() == (None, True)


def test_bad_requests_change_nothing() -> None:
    controls = EpisodeControls()
    assert _call(controls, SERVICE_START, b"not json")["ok"] is False
    assert _call(controls, SERVICE_START, [1, 2])["ok"] is False
    assert _call(controls, SERVICE_START, {"task": ""})["ok"] is False
    assert _call(controls, SERVICE_START, {"task": 5})["ok"] is False
    controls.set_phase(RECORDING, 0)
    assert _call(controls, SERVICE_FINISH, {"outcome": "maybe"})["ok"] is False
    assert _call(controls, "/recorder/nope", {})["ok"] is False
    assert not controls.start and controls.end_outcome is None and not controls.discard
    assert _call(controls, SERVICE_QUIT, {})["ok"] and controls.stop


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_hosts_are_allowed(host: str) -> None:
    require_loopback(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.20", "::", "robot.local"])
def test_other_hosts_are_refused(host: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        require_loopback(host)


def test_loop_idles_until_started_and_records_what_the_app_says(tmp_path: Path) -> None:
    time = FakeTime()
    robot = FakeFollower(time, cameras=("wrist",))
    controls = EpisodeControls()
    log: list[tuple[int, str]] = []

    def drive(call: int) -> None:
        log.append((call, controls.phase))
        if call == 5:
            assert controls.phase == IDLE
            controls.request_finish(SUCCESS)  # nothing to finish: must not hit episode 0
        if call == 40:
            assert controls.phase == IDLE and controls.saved == 0, "recorded without a start"
            controls.request_start("first task")
        if call == 60:
            assert controls.phase == RECORDING
            controls.request_finish(FAILURE)
        if call == 100:
            controls.request_start()
        if call == 120:
            controls.request_discard()
        if call == 160:
            controls.request_start()
        if call == 180:
            controls.request_stop()

    leader = FakeLeader(time, on_tick=drive)
    with _session(tmp_path, robot, leader) as session:
        results = record_episodes(
            robot,
            leader,
            session,
            task="default task",
            num_episodes=5,
            episode_time_s=0,
            reset_time_s=0,
            start_index=0,
            wait_for_start=True,
            controls=controls,
            clock=time.clock(),
            say=lambda _message: None,
        )
    # Failure kept, discard gone, stop keeps the running episode as a success.
    assert [(r.episode_index, r.outcome) for r in results] == [(0, FAILURE), (1, SUCCESS)]
    assert controls.phase == "stopped"
    assert controls.saved == 2 and controls.kept == 1
    tasks = [_task(result.path) for result in results]
    assert tasks == ["first task", "first task"], "a start's task sticks for later episodes"
    assert sorted(p.name for p in (tmp_path / "episodes").glob("*.mcap")) == [
        "episode_0000_test.mcap",
        "episode_0001_test.mcap",
    ]


def _task(path: Path) -> str:
    with path.open("rb") as handle:
        for record in make_reader(handle).iter_metadata():
            if record.name == "episode":
                return record.metadata["task"]
    raise AssertionError("no episode metadata")


def test_keys_mode_is_unchanged_and_serves_no_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    time = FakeTime()
    robot, leader = FakeFollower(time, cameras=("wrist",)), FakeLeader(time)
    monkeypatch.setattr(cli, "record_episodes", _with_fake_clock(cli.record_episodes, time))
    servers = []
    real = cli.start_stream

    def spy(cfg, context, controls=None):
        assert controls is None
        server = real(cfg, context)
        servers.append(server)
        return server

    monkeypatch.setattr(cli, "start_stream", spy)
    results = cli.run(
        _streaming_config(tmp_path, num_episodes=2),
        make_robot=lambda _cfg: robot,
        make_teleop=lambda _cfg: leader,
        listen_keys=False,
    )
    assert len(results) == 2 and servers


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"control": "foxglove", "stream": False}, "needs the stream"),
        ({"control": "foxglove", "stream_host": "0.0.0.0"}, "loopback"),
        ({"control": "both", "stream_host": "192.168.1.5"}, "loopback"),
        ({"control": "mouse"}, "--control must be"),
    ],
)
def test_unsafe_or_invalid_control_settings_fail_before_connecting(
    tmp_path: Path, overrides: dict, message: str
) -> None:
    time = FakeTime()
    robot = FakeFollower(time, cameras=("wrist",))

    def make_robot(_cfg):
        raise AssertionError("must fail before any device is created")

    with pytest.raises(ValueError, match=message):
        cli.run(
            _streaming_config(tmp_path, **overrides),
            make_robot=make_robot,
            make_teleop=lambda _cfg: FakeLeader(time),
            listen_keys=False,
        )
    assert not robot.is_connected


def test_foxglove_client_starts_and_finishes_an_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole path over a real WebSocket: the app's service calls start and
    finish the episode, the status topic follows along, and the status topic
    stays out of the MCAP."""
    time = FakeTime()
    robot = FakeFollower(time, cameras=("wrist",))
    clients: list[FoxgloveTestClient] = []
    responses: dict[str, dict] = {}
    progress = {"started_at": None}

    def drive(call: int) -> None:
        [client] = clients
        if progress["started_at"] is None:
            if call == 3:
                responses["early_finish"] = client.call_service(SERVICE_FINISH, {})
            if call == 5:
                # The fake clock never sleeps; let the 4 Hz publisher show "idle".
                wall.sleep(0.6)
                responses["start"] = client.call_service(SERVICE_START, {"task": "from the app"})
                progress["started_at"] = call
        elif call == progress["started_at"] + 20:
            # Let the 4 Hz publisher show "recording" before the episode ends.
            wall.sleep(0.6)
            responses["finish"] = client.call_service(SERVICE_FINISH, {"outcome": "failure"})

    leader = FakeLeader(time, on_tick=drive)
    monkeypatch.setattr(cli, "record_episodes", _with_fake_clock(cli.record_episodes, time))
    real = cli.start_stream

    def start_and_connect(cfg, context, controls=None):
        server = real(cfg, context, controls)
        client = FoxgloveTestClient(server.port)
        client.subscribe([TOPIC_STATUS, "/observation/state"])
        clients.append(client)
        wall.sleep(0.3)
        return server

    monkeypatch.setattr(cli, "start_stream", start_and_connect)
    [path] = [
        r.path
        for r in cli.run(
            _streaming_config(
                tmp_path, control="foxglove", episode_time_s=0, num_episodes=1, stream_port=0
            ),
            make_robot=lambda _cfg: robot,
            make_teleop=lambda _cfg: leader,
            listen_keys=False,
        )
    ]
    [client] = clients
    client.closed.wait(timeout=5)
    client.close()

    assert responses["early_finish"]["ok"] is False
    assert responses["start"] == {"ok": True, "state": IDLE}
    assert responses["finish"] == {"ok": True, "state": RECORDING}
    assert _task(path) == "from the app"
    with path.open("rb") as handle:
        reader = make_reader(handle)
        topics = {channel.topic for channel in reader.get_summary().channels.values()}
        metadata = {m.name: m.metadata for m in reader.iter_metadata()}
    assert metadata["recording"]["outcome"] == FAILURE
    assert TOPIC_STATUS not in topics and "/observation/state" in topics

    # The server stops right after the last episode, so the final "stopped" may not arrive.
    states = [json.loads(raw)["state"] for raw in client.messages[TOPIC_STATUS]]
    assert states[0] == "idle" and "recording" in states

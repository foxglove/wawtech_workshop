"""Teleop control loop that records episodes through a :class:`RecordingSession`.

Each tick reads the follower observation, reads the leader action, sends that
action to the follower, and logs state, action and camera images with one wall
clock timestamp. As in ``lerobot-record``, the recorded action is the leader's
action before the follower clips it with ``max_relative_target``.

Episode flow: record for up to ``episode_time_s`` (0 = no limit, end it with a key), then
teleop without recording for ``reset_time_s`` while the scene is reset (``n`` ends the
reset early). Keys (focus the
terminal, or any window when pynput can capture):

* ``n`` or Right: end the episode, outcome ``success``
* ``f``: end the episode, outcome ``failure``
* ``r`` or Left: discard the episode and record it again
* ``q`` or Esc: stop after this episode (it is kept with outcome ``success``
  unless it was discarded)
* ``s``: start the next episode (only when the loop waits for a start)

With ``wait_for_start`` the loop does not roll into the next episode by itself:
after each episode (and its reset) it teleops without recording until a start
arrives, from ``s`` or from the Foxglove ``/recorder/start`` service.

An episode that runs out of time gets ``timeout_outcome``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from wawtech_workshop.recorder.session import EpisodeResult, RecordingSession

logger = logging.getLogger(__name__)

SUCCESS = "success"
FAILURE = "failure"
CONTROLS_HELP = "n=success, f=failure, r=discard and redo, q=stop"
CONTROLS_HELP_WAITING = "s=start, " + CONTROLS_HELP

IDLE = "idle"
RECORDING = "recording"
RESETTING = "resetting"
STOPPED = "stopped"


class Follower(Protocol):
    name: str

    @property
    def observation_features(self) -> Mapping[str, Any]: ...

    @property
    def action_features(self) -> Mapping[str, Any]: ...

    def get_observation(self) -> Mapping[str, Any]: ...

    def send_action(self, action: Mapping[str, Any]) -> Mapping[str, Any]: ...


class Leader(Protocol):
    @property
    def action_features(self) -> Mapping[str, Any]: ...

    def get_action(self) -> Mapping[str, Any]: ...


@dataclass
class EpisodeControls:
    """Flags and phase shared between the control loop and whatever drives it.

    The key listener thread and the Foxglove service handlers set flags; the
    control loop reads them and publishes the phase for the status topic.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    end_outcome: str | None = None
    discard: bool = False
    stop: bool = False
    start: bool = False
    # Task for the next episodes when a start request names one.
    task: str | None = None
    # Published by the loop.
    phase: str = IDLE
    episode_index: int | None = None
    phase_since: float = field(default_factory=time.monotonic)
    saved: int = 0
    kept: int = 0
    message: str = ""

    def on_key(self, name: str) -> None:
        key = name.lower()
        with self._lock:
            if key in ("n", "right"):
                self.end_outcome = SUCCESS
            elif key == "f":
                self.end_outcome = FAILURE
            elif key in ("r", "left"):
                self.discard = True
            elif key in ("q", "esc"):
                self.stop = True
            elif key == "s":
                self.start = True

    def request_start(self, task: str | None = None) -> None:
        with self._lock:
            if task is not None:
                self.task = task
            self.start = True

    def request_finish(self, outcome: str) -> None:
        with self._lock:
            self.end_outcome = outcome

    def request_discard(self) -> None:
        with self._lock:
            self.discard = True

    def request_stop(self) -> None:
        with self._lock:
            self.stop = True

    def episode_should_end(self) -> bool:
        with self._lock:
            return self.end_outcome is not None or self.discard or self.stop

    def start_should_begin(self) -> bool:
        with self._lock:
            return self.start or self.stop

    def take_start(self) -> None:
        with self._lock:
            self.start = False

    def take_episode_end(self) -> tuple[str | None, bool]:
        """Return and clear (outcome, discard). ``stop`` stays set."""
        with self._lock:
            outcome, discard = self.end_outcome, self.discard
            self.end_outcome = None
            self.discard = False
            return outcome, discard

    def set_phase(self, phase: str, episode_index: int | None = None) -> None:
        with self._lock:
            self.phase = phase
            self.episode_index = episode_index
            self.phase_since = time.monotonic()

    def note(self, message: str, *, saved: bool = False, kept: bool = False) -> None:
        with self._lock:
            self.message = message
            self.saved += saved
            self.kept += kept

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self.phase,
                "episode_index": self.episode_index,
                "elapsed_s": round(time.monotonic() - self.phase_since, 1),
                "task": self.task,
                "saved": self.saved,
                "kept": self.kept,
                "message": self.message,
            }


def state_names_for(robot: Follower, teleop: Leader) -> tuple[str, ...]:
    """Joint labels in the follower's order; the leader must send the same keys."""
    names = tuple(robot.action_features)
    leader = set(teleop.action_features)
    if set(names) != leader:
        raise ValueError(
            "Leader and follower joints differ: follower "
            f"{sorted(names)}, leader {sorted(leader)}. The converter needs identical labels "
            "on /observation/state and /action/state."
        )
    observed = set(robot.observation_features)
    missing = [name for name in names if name not in observed]
    if missing:
        raise ValueError(f"Follower observation has no values for {missing}.")
    return names


def camera_names_for(robot: Follower) -> tuple[str, ...]:
    """RGB cameras (``H x W x 3`` observation features). Depth streams are skipped."""
    names = tuple(
        key
        for key, feature in robot.observation_features.items()
        if isinstance(feature, tuple) and len(feature) == 3 and feature[2] == 3
    )
    if not names:
        raise ValueError(
            "The follower has no RGB cameras. Configure --robot.cameras; the converter "
            "needs at least one /observation/images/<camera> topic."
        )
    return names


@dataclass
class LoopClock:
    """Injectable clock so tests can run the loop without sleeping."""

    wall_ns: Callable[[], int] = time.time_ns
    monotonic: Callable[[], float] = time.perf_counter
    sleep: Callable[[float], None] = time.sleep


def run_phase(
    robot: Follower,
    teleop: Leader,
    session: RecordingSession,
    *,
    fps: int,
    duration_s: float,
    controls: EpisodeControls,
    clock: LoopClock,
    should_end: Callable[[], bool] | None = None,
) -> int:
    """Teleop at ``fps`` for ``duration_s`` (0 = until ``should_end``). Returns the ticks.

    ``should_end`` defaults to :meth:`EpisodeControls.episode_should_end`.

    Every tick is logged; it reaches the MCAP only while an episode is open,
    and the live view whenever one is connected.
    """
    should_end = should_end or controls.episode_should_end
    period = 1.0 / fps
    start = clock.monotonic()
    ticks = 0
    late = 0
    while True:
        tick_start = clock.monotonic()
        if duration_s > 0 and tick_start - start >= duration_s:
            break
        if should_end():
            break
        observation = robot.get_observation()
        stamp = clock.wall_ns()
        action = teleop.get_action()
        robot.send_action(action)
        session.log_step(
            stamp,
            state=observation,
            action=action,
            images={name: as_rgb(observation[name]) for name in session.camera_names},
        )
        ticks += 1
        remaining = period - (clock.monotonic() - tick_start)
        if remaining > 0:
            clock.sleep(remaining)
        else:
            late += 1
    if late:
        logger.warning("%d of %d ticks overran the %d Hz budget.", late, ticks, fps)
    return ticks


def as_rgb(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Camera frame must be HxWx3 uint8 RGB, got {array.shape} {array.dtype}.")
    return array


def record_episodes(
    robot: Follower,
    teleop: Leader,
    session: RecordingSession,
    *,
    task: str,
    num_episodes: int,
    episode_time_s: float,
    reset_time_s: float,
    start_index: int,
    timeout_outcome: str = SUCCESS,
    wait_for_start: bool = False,
    controls: EpisodeControls | None = None,
    clock: LoopClock | None = None,
    say: Callable[[str], None] = print,
) -> list[EpisodeResult]:
    """Record up to ``num_episodes`` kept episodes. Discarded ones are recorded again.

    With ``wait_for_start`` every episode, the first included, waits in an idle
    phase (teleop, not recording) until ``controls`` gets a start request.
    """
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}.")
    controls = controls or EpisodeControls()
    clock = clock or LoopClock()
    help_text = CONTROLS_HELP_WAITING if wait_for_start else CONTROLS_HELP
    results: list[EpisodeResult] = []
    index = start_index

    def phase(name: str, duration_s: float, should_end: Callable[[], bool] | None = None) -> None:
        controls.set_phase(name, index)
        run_phase(
            robot,
            teleop,
            session,
            fps=session.fps,
            duration_s=duration_s,
            controls=controls,
            clock=clock,
            should_end=should_end,
        )

    try:
        while len(results) < num_episodes and not controls.stop:
            if wait_for_start:
                say(f"Ready for episode {index}: waiting for a start ({help_text}).")
                controls.take_start()
                phase(IDLE, 0, controls.start_should_begin)
                # Finish and discard requests made while idle must not end the episode.
                controls.take_episode_end()
                if controls.stop:
                    break
                controls.take_start()
            current_task = controls.task or task
            path = session.start_episode(task=current_task, episode_index=index)
            say(f"Recording episode {index} -> {path.name} ({help_text})")
            try:
                phase(RECORDING, episode_time_s)
            except BaseException:
                session.discard_episode()
                raise
            outcome, discard = controls.take_episode_end()
            if discard:
                session.discard_episode()
                message = f"Discarded episode {index}; recording it again."
                say(message)
                controls.note(message)
            else:
                if outcome is None:
                    outcome = SUCCESS if controls.stop else timeout_outcome
                result = session.finish_episode(outcome)
                results.append(result)
                message = (
                    f"Saved episode {index}: {result.outcome}, {result.steps} steps, "
                    f"{result.duration_s:.1f} s."
                )
                say(message)
                controls.note(message, saved=True, kept=result.outcome == SUCCESS)
                index += 1
            if controls.stop or len(results) >= num_episodes or reset_time_s <= 0:
                continue
            say(f"Reset the scene ({reset_time_s:.0f} s, not recorded).")
            phase(RESETTING, reset_time_s)
            # Any of n/f/r ends the reset early; it must not also end the next episode.
            controls.take_episode_end()
    finally:
        controls.set_phase(STOPPED, None)
    return results

"""Fake follower, leader and clock for recorder tests. No motors, cameras or sleeping."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from wawtech_workshop.recorder.loop import LoopClock

JOINTS = ("shoulder_pan", "elbow_flex", "gripper")
HEIGHT = 48
WIDTH = 64


class FakeTime:
    """Monotonic time that only moves when the loop sleeps or a device read costs time."""

    def __init__(self, start_s: float = 1_700_000_000.0, read_cost_s: float = 0.002) -> None:
        self.now = start_s
        self.read_cost_s = read_cost_s

    def clock(self) -> LoopClock:
        return LoopClock(
            wall_ns=lambda: int(round(self.now * 1_000_000_000)),
            monotonic=lambda: self.now,
            sleep=self.sleep,
        )

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, seconds)

    def spend(self) -> None:
        self.now += self.read_cost_s


def joint_value(joint_index: int, t_s: float, offset: float = 0.0) -> float:
    return float(10.0 * np.sin(0.5 * t_s + joint_index) + offset)


def camera_frame(tick: int, channel: int) -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:, :, channel] = np.uint8((tick * 9) % 256)
    frame[:, :, (channel + 1) % 3] = np.linspace(0, 255, WIDTH, dtype=np.uint8)[None, :]
    return frame


@dataclass
class FakeFollower:
    time: FakeTime
    cameras: tuple[str, ...] = ("front", "wrist")
    joints: tuple[str, ...] = JOINTS
    with_depth: bool = False
    name: str = "rebot_b601_follower"
    is_connected: bool = False
    ticks: int = 0
    sent: list[dict[str, float]] = field(default_factory=list)

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in self.joints}

    @property
    def observation_features(self) -> dict[str, object]:
        features: dict[str, object] = dict(self.action_features)
        for camera in self.cameras:
            features[camera] = (HEIGHT, WIDTH, 3)
            if self.with_depth:
                features[f"{camera}_depth"] = (HEIGHT, WIDTH, 1)
        return features

    def connect(self) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def get_observation(self) -> dict[str, object]:
        self.time.spend()
        t_s = self.time.now
        observation: dict[str, object] = {
            f"{joint}.pos": joint_value(index, t_s) for index, joint in enumerate(self.joints)
        }
        for index, camera in enumerate(self.cameras):
            observation[camera] = camera_frame(self.ticks, index % 3)
            if self.with_depth:
                observation[f"{camera}_depth"] = np.zeros((HEIGHT, WIDTH, 1), dtype=np.uint16)
        self.ticks += 1
        return observation

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.sent.append(dict(action))
        return dict(action)


@dataclass
class FakeLeader:
    time: FakeTime
    joints: tuple[str, ...] = JOINTS
    is_connected: bool = False
    calls: int = 0
    # Called with the number of get_action calls so far; tests press keys here.
    on_tick: Callable[[int], None] | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in self.joints}

    def connect(self) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def get_action(self) -> dict[str, float]:
        self.calls += 1
        if self.on_tick is not None:
            self.on_tick(self.calls)
        t_s = self.time.now
        return {
            f"{joint}.pos": joint_value(index, t_s, offset=1.0)
            for index, joint in enumerate(self.joints)
        }

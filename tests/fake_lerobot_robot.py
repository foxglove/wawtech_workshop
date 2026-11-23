"""A fake robot registered with LeRobot, so ``lerobot-rollout`` machinery runs offline.

``--robot.type=wawtech_fake`` builds this robot through LeRobot's own
``make_robot_from_config`` (``WawtechFakeRobotConfig`` -> ``WawtechFakeRobot``, found in
this module). It has three ``.pos`` joints and one 64x48 RGB camera named ``wrist``, and
it keeps every action it was sent. Joints move to the last command, like a stiff arm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from lerobot.robots import Robot, RobotConfig

JOINTS = ("shoulder_pan", "elbow_flex", "gripper")
CAMERA = "wrist"
HEIGHT = 48
WIDTH = 64


@RobotConfig.register_subclass("wawtech_fake")
@dataclass(kw_only=True)
class WawtechFakeRobotConfig(RobotConfig):
    pass


class WawtechFakeRobot(Robot):
    config_class = WawtechFakeRobotConfig
    name = "wawtech_fake"
    # Every robot built since the last reset, so a test can look at what it was sent.
    built: ClassVar[list[WawtechFakeRobot]] = []

    def __init__(self, config: WawtechFakeRobotConfig) -> None:
        super().__init__(config)
        self._connected = False
        self._position = {f"{joint}.pos": 0.0 for joint in JOINTS}
        self._ticks = 0
        self.sent: list[dict[str, float]] = []
        self.observations = 0
        WawtechFakeRobot.built.append(self)

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in JOINTS}

    @property
    def observation_features(self) -> dict[str, Any]:
        return {**self.action_features, CAMERA: (HEIGHT, WIDTH, 3)}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        self._connected = True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        self.observations += 1
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:, :, self._ticks % 3] = (self._ticks * 7) % 256
        frame[:, :, (self._ticks + 1) % 3] = np.linspace(0, 255, WIDTH, dtype=np.uint8)[None, :]
        self._ticks += 1
        return {**self._position, CAMERA: frame}

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        applied = {key: float(value) for key, value in action.items()}
        self.sent.append(applied)
        self._position.update(applied)
        return applied

    def disconnect(self) -> None:
        self._connected = False

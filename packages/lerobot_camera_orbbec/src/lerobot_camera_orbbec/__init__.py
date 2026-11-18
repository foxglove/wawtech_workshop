"""Orbbec camera plugin for LeRobot.

Importing this package registers the ``orbbec`` camera config. The device class
is exported here so LeRobot's factory can find ``OrbbecCamera`` on the config's
parent module.
"""

from .camera_orbbec import OrbbecCamera
from .configuration_orbbec import OrbbecCameraConfig

__all__ = ["OrbbecCamera", "OrbbecCameraConfig"]

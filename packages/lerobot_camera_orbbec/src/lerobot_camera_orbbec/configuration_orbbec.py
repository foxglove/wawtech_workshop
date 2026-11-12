"""Configuration for an Orbbec color camera."""

from dataclasses import dataclass

from lerobot.cameras import CameraConfig, ColorMode


@CameraConfig.register_subclass("orbbec")
@dataclass(kw_only=True)
class OrbbecCameraConfig(CameraConfig):
    """Settings for one Orbbec color stream.

    ``fps``, ``width``, and ``height`` must all be set or all left as ``None``.
    ``None`` asks the SDK for its default color profile. Which profile a Gemini
    305 actually exposes is unverified on hardware.

    With the default ``color_mode`` of RGB, frames returned to LeRobot are
    uint8 arrays of shape ``(height, width, 3)``. The ``--robot.cameras`` key
    is the image topic and feature name, so it must equal ``<name>`` in
    ``/observation/images/<name>``.

    Attributes:
        serial_number_or_name: Device serial number or unique device name.
            ``None`` opens the SDK's default device.
        color_mode: Channel order of frames returned by the camera.
        warmup_s: Seconds to read frames during ``connect`` when warmup is enabled.
        timeout_ms: Default wait for a frame, in milliseconds.
    """

    serial_number_or_name: str | None = None
    color_mode: ColorMode = ColorMode.RGB
    warmup_s: float = 1.0
    timeout_ms: float = 2000

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        values = (self.fps, self.width, self.height)
        if any(value is not None for value in values) and any(value is None for value in values):
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, "
                "or none of them."
            )
        if self.fps is not None and self.fps <= 0:
            raise ValueError(f"`fps` must be positive, got {self.fps}.")
        if self.width is not None and self.width <= 0:
            raise ValueError(f"`width` must be positive, got {self.width}.")
        if self.height is not None and self.height <= 0:
            raise ValueError(f"`height` must be positive, got {self.height}.")
        if self.warmup_s < 0:
            raise ValueError(f"`warmup_s` must be >= 0, got {self.warmup_s}.")
        if self.timeout_ms <= 0:
            raise ValueError(f"`timeout_ms` must be positive, got {self.timeout_ms}.")

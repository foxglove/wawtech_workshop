"""Orbbec color camera for LeRobot.

The background read thread keeps the latest frame behind a lock, matching the
structure of LeRobot's RealSense camera. ``pyorbbecsdk`` is imported only inside
the backend methods that talk to a device.

Behaviour on a real Gemini 305, and on macOS, is UNVERIFIED. Method names match
the type stubs shipped in ``pyorbbecsdk2==2.1.3`` (``Pipeline``, ``Config``,
``Context.query_devices``, ``OBSensorType.COLOR_SENSOR``, ``VideoFrame.get_data``).
No capture from this workshop's hardware was run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import Event, Lock, Thread, current_thread
from typing import Any, Protocol

import numpy as np
from lerobot.cameras import ColorMode
from lerobot.cameras.camera import Camera
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from numpy.typing import NDArray

from .configuration_orbbec import OrbbecCameraConfig

logger = logging.getLogger(__name__)

_PREFERRED_FORMATS = ("RGB", "BGR", "MJPG", "YUYV")
_SDK_INSTALL_HINT = (
    "pyorbbecsdk is not installed. Install it with `uv add pyorbbecsdk2==2.1.3` "
    "(the distribution name is pyorbbecsdk2; the import name is pyorbbecsdk)."
)


def _import_pyorbbecsdk() -> Any:
    try:
        import pyorbbecsdk
    except ImportError as exc:
        raise ImportError(_SDK_INSTALL_HINT) from exc
    return pyorbbecsdk


@dataclass(frozen=True)
class _StreamProfile:
    width: int
    height: int
    fps: int
    format_name: str

    def describe(self) -> str:
        return f"{self.width}x{self.height}@{self.fps} {self.format_name}"


def _select_color_profile(
    profiles: list[_StreamProfile],
    *,
    width: int,
    height: int,
    fps: int,
) -> _StreamProfile:
    """Pick a supported color profile at the requested size and rate.

    Preference order is RGB, BGR, MJPG, then YUYV. A requested mode that the
    device does not list raises ``ValueError`` naming every reported profile.
    """
    matches = [
        profile
        for profile in profiles
        if profile.width == width and profile.height == height and profile.fps == fps
    ]
    supported = [profile for profile in matches if profile.format_name in _PREFERRED_FORMATS]
    if not supported:
        listing = ", ".join(profile.describe() for profile in profiles) or "(none)"
        raise ValueError(
            f"No supported color profile {width}x{height}@{fps}. Available profiles: {listing}."
        )
    rank = {name: index for index, name in enumerate(_PREFERRED_FORMATS)}
    supported.sort(key=lambda profile: rank[profile.format_name])
    return supported[0]


def _format_name(fmt: object) -> str:
    name = getattr(fmt, "name", None)
    if isinstance(name, str) and name:
        return name
    text = str(fmt)
    if "." in text:
        return text.rsplit(".", 1)[-1]
    return text


def _decode_color_buffer(
    data: bytes | np.ndarray,
    format_name: str,
    width: int,
    height: int,
) -> NDArray[np.uint8]:
    """Turn one SDK color buffer into an RGB uint8 array of shape ``(height, width, 3)``.

    ``cv2`` is used for MJPG and YUYV. OpenCV is already a dependency of
    ``lerobot`` (``opencv-python-headless``), so this plugin does not declare it again.
    """
    import cv2

    fmt = format_name.upper()
    if isinstance(data, np.ndarray):
        buffer = np.ascontiguousarray(data, dtype=np.uint8).reshape(-1)
    else:
        # frombuffer views the bytes; copy so OpenCV can write into MJPG buffers.
        buffer = np.frombuffer(data, dtype=np.uint8).copy()
    if fmt == "RGB":
        image = buffer.reshape((height, width, 3))
        return np.ascontiguousarray(image)
    if fmt == "BGR":
        image = buffer.reshape((height, width, 3))
        return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if fmt == "MJPG":
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("Failed to decode an MJPG color frame.")
        if decoded.shape[1] != width or decoded.shape[0] != height:
            raise RuntimeError(
                f"MJPG frame is {decoded.shape[1]}x{decoded.shape[0]}, expected {width}x{height}."
            )
        return np.ascontiguousarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
    if fmt == "YUYV":
        yuyv = buffer.reshape((height, width, 2))
        # YUYV and the OpenCV YUY2 fourcc are the same packing.
        return np.ascontiguousarray(cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUY2))
    raise RuntimeError(f"Unsupported Orbbec color format {format_name!r}.")


class _FrameSource(Protocol):
    def connect(self) -> tuple[int, int, int]:
        """Start streaming. Returns the actual ``(width, height, fps)``."""

    def read_frame(self, timeout_ms: float) -> NDArray[np.uint8]:
        """Return one RGB uint8 frame of shape ``(height, width, 3)``."""

    def disconnect(self) -> None:
        """Release the device."""


class _OrbbecBackend:
    """Small adapter around ``pyorbbecsdk`` so tests can substitute a fake.

    Method names match the ``pyorbbecsdk2==2.1.3`` stubs. Unverified on hardware.
    """

    def __init__(self, config: OrbbecCameraConfig) -> None:
        self._config = config
        self._pipeline: Any = None
        self._width = 0
        self._height = 0
        self._fps = 0
        self._format_name = "RGB"

    def connect(self) -> tuple[int, int, int]:
        sdk = _import_pyorbbecsdk()
        pipeline = _open_pipeline(sdk, self._config.serial_number_or_name)
        try:
            profile = _enable_color_stream(sdk, pipeline, self._config)
            stream_config = sdk.Config()
            stream_config.enable_stream(profile.sdk_profile)
            pipeline.start(stream_config)
        except Exception:
            try:
                pipeline.stop()
            except Exception:
                logger.debug(
                    "Ignoring pipeline.stop() after a failed Orbbec connect.", exc_info=True
                )
            raise
        self._pipeline = pipeline
        self._width = profile.width
        self._height = profile.height
        self._fps = profile.fps
        self._format_name = profile.format_name
        return profile.width, profile.height, profile.fps

    def read_frame(self, timeout_ms: float) -> NDArray[np.uint8]:
        if self._pipeline is None:
            raise RuntimeError("Orbbec pipeline is not started.")
        frames = self._pipeline.wait_for_frames(max(1, int(timeout_ms)))
        if frames is None:
            raise TimeoutError(f"Orbbec wait_for_frames returned no frames within {timeout_ms} ms.")
        color = frames.get_color_frame()
        if color is None:
            raise RuntimeError("Orbbec frameset did not contain a color frame.")
        width = int(color.get_width())
        height = int(color.get_height())
        format_name = _format_name(color.get_format())
        return _decode_color_buffer(color.get_data(), format_name, width, height)

    def disconnect(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        if pipeline is not None:
            pipeline.stop()


@dataclass
class _SdkProfile:
    width: int
    height: int
    fps: int
    format_name: str
    sdk_profile: Any


def _open_pipeline(sdk: Any, serial_number_or_name: str | None) -> Any:
    if serial_number_or_name is None:
        return sdk.Pipeline()
    context = sdk.Context()
    devices = context.query_devices()
    try:
        device = devices.get_device_by_serial_number(serial_number_or_name)
        return sdk.Pipeline(device)
    except Exception:
        logger.debug("No Orbbec device with serial %s; trying device name.", serial_number_or_name)
    matches: list[Any] = []
    names: list[str] = []
    for index in range(int(devices.get_count())):
        device = devices.get_device_by_index(index)
        info = device.get_device_info()
        name = str(info.get_name())
        serial = str(info.get_serial_number())
        names.append(f"{name} ({serial})")
        if name == serial_number_or_name or serial == serial_number_or_name:
            matches.append(device)
    if len(matches) == 1:
        return sdk.Pipeline(matches[0])
    if not matches:
        raise ValueError(
            f"No Orbbec camera found with serial or name {serial_number_or_name!r}. "
            f"Available cameras: {names or '(none)'}."
        )
    raise ValueError(
        f"Multiple Orbbec cameras match {serial_number_or_name!r}. Pass a unique serial number."
    )


def _enable_color_stream(sdk: Any, pipeline: Any, config: OrbbecCameraConfig) -> _SdkProfile:
    profile_list = pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
    collected: list[_SdkProfile] = []
    for index in range(int(profile_list.get_count())):
        raw = profile_list.get_stream_profile_by_index(index)
        video = raw.as_video_stream_profile()
        collected.append(
            _SdkProfile(
                width=int(video.get_width()),
                height=int(video.get_height()),
                fps=int(video.get_fps()),
                format_name=_format_name(video.get_format()),
                sdk_profile=video,
            )
        )
    described = [
        _StreamProfile(item.width, item.height, item.fps, item.format_name) for item in collected
    ]
    if config.width is None or config.height is None or config.fps is None:
        default = profile_list.get_default_video_stream_profile()
        chosen = _SdkProfile(
            width=int(default.get_width()),
            height=int(default.get_height()),
            fps=int(default.get_fps()),
            format_name=_format_name(default.get_format()),
            sdk_profile=default,
        )
        if chosen.format_name not in _PREFERRED_FORMATS:
            listing = ", ".join(profile.describe() for profile in described) or "(none)"
            raise ValueError(
                f"Default color profile {chosen.width}x{chosen.height}@{chosen.fps} "
                f"{chosen.format_name} is not a supported format. Available profiles: {listing}."
            )
        return chosen
    selected = _select_color_profile(
        described, width=config.width, height=config.height, fps=config.fps
    )
    for item in collected:
        if (
            item.width == selected.width
            and item.height == selected.height
            and item.fps == selected.fps
            and item.format_name == selected.format_name
        ):
            return item
    raise RuntimeError(f"Selected color profile {selected.describe()} disappeared before enable.")


def _query_orbbec_cameras() -> list[dict[str, Any]]:
    """Enumerate cameras. Shape matches LeRobot's RealSense ``find_cameras`` entries."""
    sdk = _import_pyorbbecsdk()
    context = sdk.Context()
    devices = context.query_devices()
    found: list[dict[str, Any]] = []
    for index in range(int(devices.get_count())):
        device = devices.get_device_by_index(index)
        info = device.get_device_info()
        camera_info: dict[str, Any] = {
            "name": str(info.get_name()),
            "type": "Orbbec",
            "id": str(info.get_serial_number()),
            "firmware_version": str(info.get_firmware_version()),
            "connection_type": str(info.get_connection_type()),
        }
        pipeline = sdk.Pipeline(device)
        try:
            profiles = pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
            default = profiles.get_default_video_stream_profile()
            camera_info["default_stream_profile"] = {
                "stream_type": "color",
                "format": _format_name(default.get_format()),
                "width": int(default.get_width()),
                "height": int(default.get_height()),
                "fps": int(default.get_fps()),
            }
        finally:
            # The pipeline was not started. stop() on an idle pipeline is untested.
            try:
                pipeline.stop()
            except Exception:
                logger.debug("Ignoring pipeline.stop() after profile enumeration.", exc_info=True)
        found.append(camera_info)
    return found


class OrbbecCamera(Camera):
    """Color camera backed by ``pyorbbecsdk``, or by an injected frame source in tests."""

    def __init__(self, config: OrbbecCameraConfig, *, backend: _FrameSource | None = None) -> None:
        super().__init__(config)
        self.config = config
        self.color_mode = config.color_mode
        self.timeout_ms = config.timeout_ms
        self.warmup_s = config.warmup_s
        self._backend = backend
        self._connected = False
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock = Lock()
        self.latest_frame: NDArray[np.uint8] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event = Event()
        self._last_read_error: BaseException | None = None

    def __str__(self) -> str:
        target = self.config.serial_number_or_name or "default"
        return f"{self.__class__.__name__}({target})"

    @property
    def is_connected(self) -> bool:
        return self._connected

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return _query_orbbec_cameras()

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        if self._backend is None:
            self._backend = _OrbbecBackend(self.config)
        try:
            width, height, fps = self._backend.connect()
        except (ImportError, ValueError, DeviceAlreadyConnectedError):
            raise
        except Exception as exc:
            raise ConnectionError(f"Failed to open {self}: {exc}") from exc
        self.width = width
        self.height = height
        self.fps = fps
        self._start_read_thread()
        self._connected = True
        if not warmup:
            logger.info("%s connected.", self)
            return
        try:
            self._run_warmup()
        except Exception:
            self.disconnect()
            raise
        logger.info("%s connected.", self)

    def _run_warmup(self) -> None:
        timeout_ms = max(self.timeout_ms, self.warmup_s * 1000.0)
        self.async_read(timeout_ms=timeout_ms)
        deadline = time.perf_counter() + self.warmup_s
        while time.perf_counter() < deadline:
            remaining_ms = (deadline - time.perf_counter()) * 1000.0
            if remaining_ms <= 0:
                break
            try:
                self.async_read(timeout_ms=remaining_ms)
            except TimeoutError:
                break
        with self.frame_lock:
            if self.latest_frame is None:
                raise ConnectionError(f"{self} failed to capture frames during warmup.")

    def _apply_color_mode(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(f"{self} expected an HxWx3 frame, got shape {frame.shape}.")
        if self.height is not None and self.width is not None:
            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                raise RuntimeError(
                    f"{self} frame is {frame.shape[1]}x{frame.shape[0]}, "
                    f"expected {self.width}x{self.height}."
                )
        if self.color_mode == ColorMode.BGR:
            return np.ascontiguousarray(frame[..., ::-1])
        if self.color_mode != ColorMode.RGB:
            raise ValueError(f"Invalid color mode {self.color_mode!r}.")
        return frame

    def _read_loop(self) -> None:
        stop_event = self.stop_event
        backend = self._backend
        if stop_event is None or backend is None:
            raise RuntimeError(f"{self}: read loop started without a backend.")
        failure_count = 0
        while not stop_event.is_set():
            try:
                frame = self._apply_color_mode(backend.read_frame(self.timeout_ms))
                captured = time.perf_counter()
                with self.frame_lock:
                    if stop_event.is_set():
                        break
                    self.latest_frame = frame
                    self.latest_timestamp = captured
                self.new_frame_event.set()
                failure_count = 0
            except DeviceNotConnectedError:
                break
            except Exception as exc:
                failure_count += 1
                logger.warning("Error reading frame in background thread for %s: %s", self, exc)
                if failure_count > 10:
                    self._last_read_error = exc
                    logger.error("%s exceeded maximum consecutive read failures.", self)
                    break

    def _read_thread_stopped(self) -> RuntimeError:
        message = f"{self} read thread is not running."
        if self._last_read_error is not None:
            message = f"{message} Last error: {self._last_read_error}"
        return RuntimeError(message)

    def _start_read_thread(self) -> None:
        self._stop_read_thread()
        self._last_read_error = None
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read_loop", daemon=True)
        self.thread.start()

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        thread = self.thread
        if thread is not None and thread.is_alive() and thread is not current_thread():
            thread.join(timeout=2.0)
            if thread.is_alive():
                logger.warning("%s read thread did not terminate within timeout.", self)
        self.thread = None
        self.stop_event = None
        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    def _async_read(self, timeout_ms: float) -> NDArray[np.uint8]:
        if self.thread is None or not self.thread.is_alive():
            raise self._read_thread_stopped()
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms."
            )
        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()
        if frame is None:
            raise RuntimeError(f"Internal error: event set but no frame available for {self}.")
        return frame

    @check_if_not_connected
    def read(self) -> NDArray[np.uint8]:
        self.new_frame_event.clear()
        return self._async_read(self.timeout_ms)

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[np.uint8]:
        return self._async_read(timeout_ms)

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[np.uint8]:
        if self.thread is None or not self.thread.is_alive():
            raise self._read_thread_stopped()
        with self.frame_lock:
            frame = self.latest_frame
            timestamp = self.latest_timestamp
        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")
        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )
        return frame

    def disconnect(self) -> None:
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._connected = False
        self._stop_read_thread()
        if self._backend is not None:
            self._backend.disconnect()
        logger.info("%s disconnected.", self)

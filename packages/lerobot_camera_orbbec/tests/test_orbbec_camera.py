"""Orbbec plugin registration and camera behaviour with a fake backend."""

from __future__ import annotations

import sys
import time

import lerobot_camera_orbbec
import numpy as np
import pytest
from lerobot.cameras import CameraConfig, ColorMode
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot_camera_orbbec.camera_orbbec import (
    OrbbecCamera,
    _decode_color_buffer,
    _select_color_profile,
    _StreamProfile,
)
from lerobot_camera_orbbec.configuration_orbbec import OrbbecCameraConfig
from numpy.typing import NDArray


class FakeBackend:
    def __init__(self, frame: NDArray[np.uint8]) -> None:
        self.frame = frame
        self.connected = False
        self.disconnects = 0

    def connect(self) -> tuple[int, int, int]:
        self.connected = True
        height, width = self.frame.shape[:2]
        return width, height, 30

    def read_frame(self, timeout_ms: float) -> NDArray[np.uint8]:
        del timeout_ms
        time.sleep(0.01)
        if not self.connected:
            raise RuntimeError("fake backend is not connected")
        return self.frame.copy()

    def disconnect(self) -> None:
        self.connected = False
        self.disconnects += 1


def _frame(height: int = 4, width: int = 6) -> NDArray[np.uint8]:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 1] = 20
    image[..., 2] = 200
    return image


def test_import_registers_orbbec_choice() -> None:
    assert lerobot_camera_orbbec.OrbbecCameraConfig is OrbbecCameraConfig
    choice = CameraConfig.get_choice_class("orbbec")
    assert choice is OrbbecCameraConfig


def test_factory_builds_orbbec_camera() -> None:
    config = OrbbecCameraConfig(width=640, height=480, fps=30, serial_number_or_name="abc")
    cameras = make_cameras_from_configs({"front": config})
    camera = cameras["front"]
    assert isinstance(camera, OrbbecCamera)
    assert camera.is_connected is False


def test_plugin_discovery_imports_distribution() -> None:
    sys.modules.pop("lerobot_camera_orbbec", None)
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    assert "lerobot_camera_orbbec" in sys.modules


def test_read_before_connect_raises() -> None:
    camera = OrbbecCamera(
        OrbbecCameraConfig(width=6, height=4, fps=30), backend=FakeBackend(_frame())
    )
    with pytest.raises(DeviceNotConnectedError):
        camera.read()


def test_fake_backend_read_color_mode_and_disconnect() -> None:
    source = _frame()
    backend = FakeBackend(source)
    camera = OrbbecCamera(
        OrbbecCameraConfig(
            width=6, height=4, fps=30, color_mode=ColorMode.BGR, warmup_s=0, timeout_ms=1000
        ),
        backend=backend,
    )
    camera.connect(warmup=False)
    try:
        frame = camera.read()
        assert frame.shape == (4, 6, 3)
        assert frame.dtype == np.uint8
        np.testing.assert_array_equal(frame[..., 0], source[..., 2])
        np.testing.assert_array_equal(frame[..., 2], source[..., 0])

        updated = source.copy()
        updated[..., 1] = 255
        backend.frame = updated
        deadline = time.perf_counter() + 1.0
        latest = frame
        while time.perf_counter() < deadline:
            latest = camera.async_read(timeout_ms=200)
            if int(latest[0, 0, 1]) == 255:
                break
        else:
            raise AssertionError("async_read did not return the latest frame")
        assert latest.shape == (4, 6, 3)
    finally:
        camera.disconnect()
    assert camera.is_connected is False
    assert camera.thread is None
    assert backend.disconnects == 1
    assert backend.connected is False


def test_config_rejects_partial_resolution() -> None:
    with pytest.raises(ValueError, match="fps"):
        OrbbecCameraConfig(fps=30, width=640)
    with pytest.raises(ValueError, match="fps"):
        OrbbecCameraConfig(height=480, fps=30)
    OrbbecCameraConfig()
    OrbbecCameraConfig(width=1280, height=720, fps=30)
    with pytest.raises(ValueError):
        OrbbecCameraConfig(width=64, height=48, fps=30, color_mode="not-a-mode")


def test_missing_profile_lists_available_modes() -> None:
    profiles = [
        _StreamProfile(640, 480, 30, "MJPG"),
        _StreamProfile(1280, 720, 15, "YUYV"),
    ]
    with pytest.raises(ValueError, match="1280x720@15 YUYV"):
        _select_color_profile(profiles, width=1280, height=720, fps=30)
    selected = _select_color_profile(
        profiles + [_StreamProfile(640, 480, 30, "RGB")],
        width=640,
        height=480,
        fps=30,
    )
    assert selected.format_name == "RGB"


def test_color_buffer_conversion_round_trip() -> None:
    import cv2

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 210
    rgb[..., 1] = 40
    rgb[..., 2] = 15
    np.testing.assert_array_equal(_decode_color_buffer(rgb.tobytes(), "RGB", 8, 8), rgb)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    np.testing.assert_array_equal(_decode_color_buffer(bgr.tobytes(), "BGR", 8, 8), rgb)
    encoded_ok, encoded = cv2.imencode(".jpg", bgr)
    assert encoded_ok
    jpeg = _decode_color_buffer(encoded.tobytes(), "MJPG", 8, 8)
    assert jpeg.shape == (8, 8, 3)
    assert jpeg[..., 0].mean() > 150
    assert jpeg[..., 2].mean() < 40
    yuyv = np.full((8, 8, 2), 128, dtype=np.uint8)
    yuv = _decode_color_buffer(yuyv.tobytes(), "YUYV", 8, 8)
    assert yuv.shape == (8, 8, 3)
    assert np.all(np.abs(yuv.astype(np.int16) - 128) < 20)


def test_read_thread_reports_the_last_backend_error() -> None:
    class FailingBackend:
        def connect(self) -> tuple[int, int, int]:
            return 6, 4, 30

        def read_frame(self, timeout_ms: float) -> NDArray[np.uint8]:
            del timeout_ms
            raise RuntimeError("sensor stalled")

        def disconnect(self) -> None:
            return None

    camera = OrbbecCamera(
        OrbbecCameraConfig(width=6, height=4, fps=30, warmup_s=0, timeout_ms=50),
        backend=FailingBackend(),
    )
    camera.connect(warmup=False)
    try:
        deadline = time.perf_counter() + 2.0
        while camera.thread is not None and camera.thread.is_alive():
            if time.perf_counter() > deadline:
                raise AssertionError("read thread did not stop after repeated failures")
            time.sleep(0.01)
        with pytest.raises(RuntimeError, match="sensor stalled"):
            camera.read()
        with pytest.raises(RuntimeError, match="read thread is not running"):
            camera.read_latest()
    finally:
        camera.disconnect()


def test_find_cameras_without_sdk_explains_install(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def broken(name: str, *args: object, **kwargs: object) -> object:
        if name == "pyorbbecsdk":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)
    with pytest.raises(ImportError, match="pyorbbecsdk2"):
        OrbbecCamera.find_cameras()

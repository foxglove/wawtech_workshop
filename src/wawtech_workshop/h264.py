"""Encode RGB frames into the H.264 stream shape the converter decodes.

Every frame becomes one Annex B access unit with SPS and PPS and no B-frames,
using :func:`wawtech_workshop.mcap_schema.h264_codec_options`. That is the
contract :func:`wawtech_workshop.dataset.video.iter_decode_h264_messages`
checks, and Foxglove plays it back as ``foxglove.CompressedVideo``.
"""

from __future__ import annotations

from fractions import Fraction

import av
import numpy as np

from wawtech_workshop.mcap_schema import h264_codec_options

_PREFERRED_ENCODERS = ("libx264", "h264", "libopenh264")


def h264_encoder_name() -> str:
    """Prefer libx264. Fall back to the first H.264 encoder this PyAV build can open."""
    last_error: Exception | None = None
    for name in _PREFERRED_ENCODERS:
        try:
            _probe_encoder(name)
        except Exception as exc:
            last_error = exc
            continue
        return name
    available = sorted(str(name) for name in av.codecs_available if "264" in str(name))
    raise RuntimeError(
        "No H.264 encoder in this PyAV build. "
        f"Tried {_PREFERRED_ENCODERS}. Candidates: {available or '(none)'}. "
        f"Last error: {last_error}"
    )


def _probe_encoder(name: str) -> None:
    codec = av.CodecContext.create(name, "w")
    codec.width = 64
    codec.height = 48
    codec.pix_fmt = "yuv420p"
    codec.time_base = Fraction(1, 30)
    codec.open()


class H264FrameEncoder:
    """Stateful encoder for one camera. ``encode`` returns one access unit per frame.

    The frame size is fixed by the first frame. H.264 with ``yuv420p`` needs even
    dimensions, so odd sizes raise.
    """

    def __init__(self, width: int, height: int, fps: int, encoder: str | None = None) -> None:
        if width % 2 or height % 2:
            raise ValueError(f"H.264 yuv420p needs even dimensions, got {width}x{height}.")
        self.width = width
        self.height = height
        self.fps = fps
        self.encoder = encoder or h264_encoder_name()
        codec = av.CodecContext.create(self.encoder, "w")
        codec.width = width
        codec.height = height
        codec.pix_fmt = "yuv420p"
        codec.time_base = Fraction(1, fps)
        codec.framerate = Fraction(fps, 1)
        codec.options = h264_codec_options(self.encoder)
        codec.open()
        self._codec = codec
        self._pts = 0

    def encode(self, rgb: np.ndarray) -> bytes:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected an HxWx3 uint8 RGB frame, got {rgb.shape} {rgb.dtype}.")
        if rgb.shape[0] != self.height or rgb.shape[1] != self.width:
            raise ValueError(
                f"Frame is {rgb.shape[1]}x{rgb.shape[0]}, encoder is {self.width}x{self.height}."
            )
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        frame.pts = self._pts
        frame.time_base = Fraction(1, self.fps)
        self._pts += 1
        data = b"".join(bytes(packet) for packet in self._codec.encode(frame))
        if not data:
            raise RuntimeError(
                f"Encoder {self.encoder} produced no packet for frame {self._pts - 1}. "
                "The zero-latency options should make every frame produce one."
            )
        return data

    def flush(self) -> bytes:
        """Drain the encoder. With the workshop options this is normally empty."""
        return b"".join(bytes(packet) for packet in self._codec.encode(None))


def encode_h264_frames(frames: list[np.ndarray], fps: int) -> list[bytes]:
    """Encode RGB frames to one Annex B blob per image, without B-frames."""
    if not frames:
        return []
    height, width = frames[0].shape[:2]
    encoder = H264FrameEncoder(width, height, fps)
    blobs = [encoder.encode(frame) for frame in frames]
    tail = encoder.flush()
    if tail:
        blobs[-1] = blobs[-1] + tail
    return blobs

"""Decode ``foxglove.CompressedVideo`` H.264 messages with PyAV.

``foxglove.datasets.video.decode_h264`` only accepts a Foxglove dataset
``EpisodeReader``. This module decodes the same message shape directly: JSON
objects whose ``data`` is base64 text or raw bytes, ``format`` is ``h264``, and
each message holds Annex B NAL units for one image with no B-frames. Keyframes
must carry SPS and PPS, which is how the synthetic writer emits them.

Alignment uses the MCAP log time supplied by the caller. The embedded
``timestamp`` is accepted but not required for ordering.
"""

from __future__ import annotations

import base64
from collections.abc import Iterable, Iterator
from fractions import Fraction
from typing import Any

import av
import numpy as np
from numpy.typing import NDArray


def _payload_bytes(data: Any) -> bytes:
    if isinstance(data, str):
        try:
            return base64.b64decode(data, validate=True)
        except Exception as exc:
            raise ValueError("CompressedVideo data is not valid base64.") from exc
    if isinstance(data, bytes | bytearray | memoryview):
        return bytes(data)
    raise ValueError(
        f"CompressedVideo data must be base64 text or bytes, got {type(data).__name__}."
    )


def decode_h264_messages(
    messages: Iterable[tuple[int, dict[str, Any]]],
) -> list[tuple[int, NDArray[np.uint8]]]:
    """Decode one ordered H.264 stream into RGB frames.

    Prefer :func:`iter_decode_h264_messages` when the caller can use one frame
    at a time. This wrapper exists for tests and other callers that need the
    full list.
    """
    return list(iter_decode_h264_messages(messages))


def iter_decode_h264_messages(
    messages: Iterable[tuple[int, dict[str, Any]]],
) -> Iterator[tuple[int, NDArray[np.uint8]]]:
    """Yield ``(log_time_ns, image)`` for one ordered H.264 stream.

    ``messages`` is ``(log_time_ns, decoded_message)`` in log-time order. Each
    message must produce exactly one image. ``image`` is a contiguous uint8
    array of shape ``(H, W, 3)``. Only the frame just yielded is retained by
    this generator.
    """
    codec = av.CodecContext.create("h264", "r")
    codec.options = {"err_detect": "explode"}
    codec.thread_count = 1
    pending: dict[int, int] = {}
    produced = 0
    message_count = 0
    try:
        for index, (log_time_ns, message) in enumerate(messages):
            message_count += 1
            if message.get("format") != "h264":
                raise ValueError(
                    "Only H.264 CompressedVideo messages are supported, "
                    f"got {message.get('format')!r}."
                )
            packet = av.Packet(_payload_bytes(message.get("data")))
            packet.pts = index
            packet.dts = index
            packet.time_base = Fraction(1, 1)
            pending[index] = log_time_ns
            frames = codec.decode(packet)
            if codec.has_b_frames:
                raise ValueError("B-frames are not supported in CompressedVideo.")
            for item in _iter_frames(frames, pending):
                produced += 1
                yield item
        if message_count:
            if codec.has_b_frames:
                raise ValueError("B-frames are not supported in CompressedVideo.")
            for item in _iter_frames(codec.decode(None), pending):
                produced += 1
                yield item
    except ValueError:
        raise
    except av.FFmpegError as exc:
        raise ValueError(f"H.264 decode failed: {exc}") from exc
    if pending or produced != message_count:
        raise ValueError(f"H.264 stream produced {produced} images for {message_count} messages.")


def _iter_frames(
    frames: list[av.VideoFrame],
    pending: dict[int, int],
) -> Iterator[tuple[int, NDArray[np.uint8]]]:
    for frame in frames:
        if frame.is_corrupt:
            raise ValueError("Decoder returned a corrupt H.264 frame.")
        if frame.pts is None or frame.pts not in pending:
            raise ValueError("Decoded image cannot be associated with one message.")
        log_time_ns = pending.pop(int(frame.pts))
        image = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Decoded frame has unexpected shape {image.shape} dtype {image.dtype}."
            )
        yield log_time_ns, image

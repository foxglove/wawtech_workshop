"""H.264 packets produced with the shared encoder options."""

from __future__ import annotations

from fractions import Fraction

import av
import numpy as np
import pytest

from tests.synthetic import encode_h264_frames, make_frame
from wawtech_workshop.dataset.video import decode_h264_messages


def test_shared_encoder_options_decode_one_image_per_packet() -> None:
    frames = [make_frame(index) for index in range(4)]
    packets = encode_h264_frames(frames, fps=30)
    decoded = decode_h264_messages(
        [(index, {"format": "h264", "data": packet}) for index, packet in enumerate(packets)]
    )
    assert len(decoded) == len(frames)
    for index, (_stamp, image) in enumerate(decoded):
        assert image.shape == frames[index].shape
        assert image.dtype == np.uint8


def test_bframes_raise() -> None:
    if "libx264" not in av.codecs_available:
        pytest.skip("libx264 is required to encode a stream that contains B-frames")
    frames = [make_frame(index) for index in range(6)]
    packets = _encode_with_bframes(frames)
    with pytest.raises(ValueError, match="B-frames are not supported"):
        decode_h264_messages(
            [(index, {"format": "h264", "data": packet}) for index, packet in enumerate(packets)]
        )


def _encode_with_bframes(frames: list[np.ndarray]) -> list[bytes]:
    """libx264 with B-frames. The shared options disable B-frames, so this does not use them."""
    codec = av.CodecContext.create("libx264", "w")
    height, width = frames[0].shape[:2]
    codec.width = width
    codec.height = height
    codec.pix_fmt = "yuv420p"
    codec.time_base = Fraction(1, 30)
    codec.framerate = Fraction(30, 1)
    codec.options = {
        "preset": "ultrafast",
        "crf": "28",
        "bf": "3",
        "g": "30",
        "x264-params": "annexb=1:repeat-headers=1:bframes=3:keyint=30",
    }
    codec.open()
    blobs: list[bytes] = []
    for index, rgb in enumerate(frames):
        video = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        video.pts = index
        pieces = [bytes(packet) for packet in codec.encode(video)]
        if pieces:
            blobs.append(b"".join(pieces))
    tail = b"".join(bytes(packet) for packet in codec.encode(None))
    if tail:
        blobs.append(tail)
    return blobs

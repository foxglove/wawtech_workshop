"""Read ``foxglove.CompressedVideo`` messages in either MCAP encoding.

The recorder writes protobuf (``foxglove.channels.CompressedVideoChannel``), which
is the only encoding Foxglove's semantic search indexes. Older recordings hold the
same fields as JSON with base64 ``data``. Both decode to one dict shape::

    {"timestamp": {"sec": int, "nsec": int}, "frame_id": str, "data": bytes, "format": str}

The Foxglove SDK can encode this message but not decode it, and the ``protobuf``
package is not a dependency, so :func:`parse_compressed_video` reads the four
fields straight from the protobuf wire format. Field numbers come from
``foxglove.CompressedVideo``: 1 timestamp, 2 frame_id, 3 data, 4 format.
"""

from __future__ import annotations

import base64
import json
from typing import Any

_VARINT, _FIXED64, _LEN, _FIXED32 = 0, 1, 2, 5


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("Truncated protobuf varint.")
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
        if shift > 63:
            raise ValueError("Protobuf varint is too long.")


def _fields(buf: bytes) -> list[tuple[int, int | bytes]]:
    """Return ``(field_number, value)`` pairs; length-delimited values are bytes."""
    out: list[tuple[int, int | bytes]] = []
    pos = 0
    while pos < len(buf):
        key, pos = _varint(buf, pos)
        number, wire = key >> 3, key & 7
        if wire == _VARINT:
            value, pos = _varint(buf, pos)
            out.append((number, value))
        elif wire == _LEN:
            length, pos = _varint(buf, pos)
            if pos + length > len(buf):
                raise ValueError("Truncated protobuf field.")
            out.append((number, buf[pos : pos + length]))
            pos += length
        elif wire == _FIXED64:
            pos += 8
        elif wire == _FIXED32:
            pos += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}.")
    return out


def parse_compressed_video(raw: bytes) -> dict[str, Any]:
    """Decode a protobuf ``foxglove.CompressedVideo`` message."""
    message: dict[str, Any] = {
        "timestamp": {"sec": 0, "nsec": 0},
        "frame_id": "",
        "data": b"",
        "format": "",
    }
    for number, value in _fields(raw):
        if number == 1 and isinstance(value, bytes):
            stamp = {"sec": 0, "nsec": 0}
            for sub, sub_value in _fields(value):
                if sub == 1 and isinstance(sub_value, int):
                    stamp["sec"] = sub_value
                elif sub == 2 and isinstance(sub_value, int):
                    stamp["nsec"] = sub_value
            message["timestamp"] = stamp
        elif number == 2 and isinstance(value, bytes):
            message["frame_id"] = value.decode("utf-8")
        elif number == 3 and isinstance(value, bytes):
            message["data"] = value
        elif number == 4 and isinstance(value, bytes):
            message["format"] = value.decode("utf-8")
    return message


def load_compressed_video(message_encoding: str, raw: bytes) -> dict[str, Any]:
    """Decode one ``foxglove.CompressedVideo`` message from a ``json`` or ``protobuf`` channel."""
    if message_encoding == "protobuf":
        return parse_compressed_video(raw)
    if message_encoding == "json":
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("CompressedVideo JSON message was not an object.")
        data = payload.get("data")
        if isinstance(data, str):
            try:
                payload["data"] = base64.b64decode(data, validate=True)
            except Exception as exc:
                raise ValueError("CompressedVideo data is not valid base64.") from exc
        return payload
    raise ValueError(
        f"CompressedVideo message encoding is {message_encoding!r}; expected 'json' or 'protobuf'."
    )

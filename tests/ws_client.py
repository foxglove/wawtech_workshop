"""A minimal Foxglove WebSocket client on the standard library, for tests.

It speaks just enough of the ``foxglove.sdk.v1`` protocol to read the channel
advertisements, subscribe, and collect message data frames.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import threading
import time
from collections import defaultdict

SUBPROTOCOL = "foxglove.sdk.v1"
MESSAGE_DATA = 0x01


class FoxgloveTestClient:
    def __init__(self, port: int, host: str = "127.0.0.1") -> None:
        self._sock = socket.create_connection((host, port), timeout=5)
        key = base64.b64encode(os.urandom(16)).decode()
        self._sock.sendall(
            (
                f"GET / HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
                f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                f"Sec-WebSocket-Version: 13\r\nSec-WebSocket-Protocol: {SUBPROTOCOL}\r\n\r\n"
            ).encode()
        )
        self._buffer = b""
        while b"\r\n\r\n" not in self._buffer:
            self._buffer += self._recv()
        head, self._buffer = self._buffer.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise ConnectionError(head.decode(errors="replace"))
        self._lock = threading.Lock()
        self.channels: dict[str, int] = {}
        self._subscriptions: dict[int, str] = {}
        self.messages: dict[str, list[bytes]] = defaultdict(list)
        self.closed = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _recv(self) -> bytes:
        data = self._sock.recv(65536)
        if not data:
            raise ConnectionError("server closed the connection")
        return data

    def _read_exact(self, size: int) -> bytes:
        while len(self._buffer) < size:
            self._buffer += self._recv()
        data, self._buffer = self._buffer[:size], self._buffer[size:]
        return data

    def _read_frame(self) -> tuple[int, bytes]:
        first, second = self._read_exact(2)
        length = second & 0x7F
        if length == 126:
            (length,) = struct.unpack("!H", self._read_exact(2))
        elif length == 127:
            (length,) = struct.unpack("!Q", self._read_exact(8))
        return first & 0x0F, self._read_exact(length)

    def _read_loop(self) -> None:
        self._sock.settimeout(None)
        try:
            while True:
                opcode, payload = self._read_frame()
                if opcode == 0x1:
                    self._on_text(json.loads(payload))
                elif opcode == 0x2 and payload[0] == MESSAGE_DATA:
                    (subscription,) = struct.unpack_from("<I", payload, 1)
                    with self._lock:
                        topic = self._subscriptions.get(subscription)
                        if topic is not None:
                            self.messages[topic].append(payload[13:])
                elif opcode == 0x8:
                    return
        except (ConnectionError, OSError, ValueError):
            return
        finally:
            self.closed.set()

    def _on_text(self, message: dict) -> None:
        if message.get("op") == "advertise":
            with self._lock:
                for channel in message["channels"]:
                    self.channels[channel["topic"]] = channel["id"]

    def _send_text(self, text: str) -> None:
        payload = text.encode()
        mask = os.urandom(4)
        header = bytes([0x81])
        if len(payload) < 126:
            header += bytes([0x80 | len(payload)])
        else:
            header += bytes([0x80 | 126]) + struct.pack("!H", len(payload))
        masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        self._sock.sendall(header + mask + masked)

    def subscribe(self, topics: list[str], timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while not all(topic in self.channels for topic in topics):
            if time.monotonic() > deadline:
                raise TimeoutError(f"not advertised: {topics} (have {sorted(self.channels)})")
            time.sleep(0.01)
        subscriptions = []
        with self._lock:
            for topic in topics:
                subscription = len(self._subscriptions) + 1
                self._subscriptions[subscription] = topic
                subscriptions.append({"id": subscription, "channelId": self.channels[topic]})
        self._send_text(json.dumps({"op": "subscribe", "subscriptions": subscriptions}))

    def close(self) -> None:
        try:
            self._sock.close()
        finally:
            self._thread.join(timeout=1)

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
SERVICE_CALL_REQUEST = 0x02
SERVICE_CALL_RESPONSE = 0x03
FETCH_ASSET_RESPONSE = 0x04


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
        self.services: dict[str, int] = {}
        self._responses: dict[int, bytes] = {}
        self._assets: dict[int, tuple[bool, bytes]] = {}
        self.capabilities: list[str] = []
        self._call_ids = 0
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
                elif opcode == 0x2 and payload[0] == FETCH_ASSET_RESPONSE:
                    request_id, status, error_length = struct.unpack_from("<IBI", payload, 1)
                    with self._lock:
                        self._assets[request_id] = (status == 0, payload[10 + error_length :])
                elif opcode == 0x2 and payload[0] == SERVICE_CALL_RESPONSE:
                    service_id, call_id, encoding_length = struct.unpack_from("<III", payload, 1)
                    with self._lock:
                        self._responses[call_id] = payload[13 + encoding_length :]
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
        if message.get("op") == "serverInfo":
            self.capabilities = list(message.get("capabilities", []))
        if message.get("op") == "advertiseServices":
            with self._lock:
                for service in message["services"]:
                    self.services[service["name"]] = service["id"]
        if message.get("op") == "advertise":
            with self._lock:
                for channel in message["channels"]:
                    self.channels[channel["topic"]] = channel["id"]

    def _send_text(self, text: str) -> None:
        self._send_frame(0x1, text.encode())

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        mask = os.urandom(4)
        header = bytes([0x80 | opcode])
        if len(payload) < 126:
            header += bytes([0x80 | len(payload)])
        else:
            header += bytes([0x80 | 126]) + struct.pack("!H", len(payload))
        masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        self._sock.sendall(header + mask + masked)

    def fetch_asset(self, uri: str, timeout: float = 5.0) -> bytes | None:
        """Ask the server for a ``package://`` asset. None if the server refuses."""
        self._call_ids += 1
        request_id = self._call_ids
        self._send_text(json.dumps({"op": "fetchAsset", "uri": uri, "requestId": request_id}))
        deadline = time.monotonic() + timeout
        while request_id not in self._assets:
            if time.monotonic() > deadline:
                raise TimeoutError(f"no response for asset {uri}")
            time.sleep(0.01)
        ok, data = self._assets.pop(request_id)
        return data if ok else None

    def call_service(self, name: str, request: dict, timeout: float = 5.0) -> dict:
        """Call a JSON service and return its decoded JSON response."""
        deadline = time.monotonic() + timeout
        while name not in self.services:
            if time.monotonic() > deadline:
                raise TimeoutError(f"service not advertised: {name} (have {sorted(self.services)})")
            time.sleep(0.01)
        self._call_ids += 1
        call_id = self._call_ids
        encoding = b"json"
        body = json.dumps(request).encode()
        self._send_frame(
            0x2,
            bytes([SERVICE_CALL_REQUEST])
            + struct.pack("<III", self.services[name], call_id, len(encoding))
            + encoding
            + body,
        )
        while call_id not in self._responses:
            if time.monotonic() > deadline:
                raise TimeoutError(f"no response from {name}")
            time.sleep(0.01)
        return json.loads(self._responses.pop(call_id))

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

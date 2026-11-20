"""Drive the recorder from the Foxglove app.

The recorder's Foxglove WebSocket server exposes four services, which the app's
Service Call panel turns into buttons, and a status topic to show what the
recorder is doing:

* ``/recorder/start`` ``{"task": "..."}`` (task optional): begin the next episode.
  Only valid while idle.
* ``/recorder/finish`` ``{"outcome": "success" | "failure"}`` (default
  ``success``): save the episode being recorded.
* ``/recorder/discard`` ``{}``: throw the episode away and go back to idle.
* ``/recorder/quit`` ``{}``: stop the session. A running episode is saved as a
  success, as with ``q`` at the keyboard.

Every response is JSON, ``{"ok": true, "state": ...}`` or
``{"ok": false, "error": ...}``. A request that does not fit the current state,
such as finishing while idle, is refused rather than queued.

``/recorder/status`` carries ``{state, episode_index, elapsed_s, task, saved,
kept, message}`` at :data:`STATUS_HZ`. It is kept out of the episode MCAPs.

The services start and stop recording only; they never command the arm. The
stream stays on loopback while they are on (see :func:`require_loopback`).
"""

from __future__ import annotations

import ipaddress
import json
import logging
import threading
from collections.abc import Callable
from typing import Any

import foxglove
from foxglove import Schema
from foxglove.websocket import Capability

from wawtech_workshop.recorder.loop import (
    FAILURE,
    IDLE,
    RECORDING,
    SUCCESS,
    EpisodeControls,
)

logger = logging.getLogger(__name__)

TOPIC_STATUS = "/recorder/status"
SERVICE_START = "/recorder/start"
SERVICE_FINISH = "/recorder/finish"
SERVICE_DISCARD = "/recorder/discard"
SERVICE_QUIT = "/recorder/quit"
STATUS_HZ = 4.0
# Topics under this prefix are for the live view and never written to episodes.
LIVE_ONLY_PREFIX = "/recorder/"

CONTROL_MODES = ("keys", "foxglove", "both")

STATUS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "wawtech.RecorderStatus",
    "properties": {
        "state": {"type": "string", "enum": ["idle", "recording", "resetting", "stopped"]},
        "episode_index": {"type": ["integer", "null"]},
        "elapsed_s": {"type": "number"},
        "task": {"type": ["string", "null"]},
        "saved": {"type": "integer"},
        "kept": {"type": "integer"},
        "message": {"type": "string"},
    },
}

_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "state": {"type": "string"},
        "error": {"type": "string"},
    },
}
_REQUEST_SCHEMAS: dict[str, dict[str, Any]] = {
    SERVICE_START: {
        "type": "object",
        "properties": {"task": {"type": "string", "description": "Overrides --task."}},
    },
    SERVICE_FINISH: {
        "type": "object",
        "properties": {"outcome": {"type": "string", "enum": [SUCCESS, FAILURE]}},
    },
    SERVICE_DISCARD: {"type": "object", "properties": {}},
    SERVICE_QUIT: {"type": "object", "properties": {}},
}


def require_loopback(host: str) -> None:
    """The control services have no authentication, so they only go on loopback."""
    if host == "localhost":
        return
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        raise ValueError(
            f"--stream_host={host} is not a loopback address. The recorder's Foxglove "
            "controls have no authentication, so they only run on 127.0.0.1. Use "
            "--control=keys to stream to another machine without them."
        )


def handle_request(controls: EpisodeControls, service: str, payload: bytes) -> dict[str, Any]:
    """Apply one service call to ``controls`` and return the JSON-able response."""
    try:
        request = json.loads(payload) if payload.strip() else {}
    except ValueError:
        return _refuse(controls, "Request body is not valid JSON.")
    if not isinstance(request, dict):
        return _refuse(controls, "Request body must be a JSON object.")
    state = controls.phase

    if service == SERVICE_START:
        if state != IDLE:
            return _refuse(controls, f"Cannot start while {state}; start works when idle.")
        task = request.get("task")
        if task is not None and (not isinstance(task, str) or not task.strip()):
            return _refuse(controls, "task must be a non-empty string.")
        controls.request_start(task.strip() if task else None)
    elif service == SERVICE_FINISH:
        if state != RECORDING:
            return _refuse(controls, f"Nothing to finish while {state}.")
        outcome = request.get("outcome", SUCCESS)
        if outcome not in (SUCCESS, FAILURE):
            return _refuse(controls, f"outcome must be {SUCCESS!r} or {FAILURE!r}.")
        controls.request_finish(outcome)
    elif service == SERVICE_DISCARD:
        if state != RECORDING:
            return _refuse(controls, f"Nothing to discard while {state}.")
        controls.request_discard()
    elif service == SERVICE_QUIT:
        controls.request_stop()
    else:
        return _refuse(controls, f"Unknown service {service}.")
    return {"ok": True, "state": state}


def _refuse(controls: EpisodeControls, error: str) -> dict[str, Any]:
    return {"ok": False, "state": controls.phase, "error": error}


def _schema(name: str, body: dict[str, Any]) -> Schema:
    return Schema(name=name, encoding="jsonschema", data=json.dumps(body).encode())


def make_services(controls: EpisodeControls) -> list[foxglove.Service]:
    def handler_for(service: str) -> Callable[[foxglove.ServiceRequest], bytes]:
        def handler(request: foxglove.ServiceRequest) -> bytes:
            response = handle_request(controls, service, request.payload)
            if not response["ok"]:
                logger.warning("%s refused: %s", service, response["error"])
            return json.dumps(response).encode()

        return handler

    services = []
    for name, request_schema in _REQUEST_SCHEMAS.items():
        title = name.rsplit("/", 1)[-1].capitalize()
        services.append(
            foxglove.Service(
                name,
                schema=foxglove.ServiceSchema(
                    name=f"wawtech.Recorder{title}",
                    request=foxglove.MessageSchema(
                        encoding="json", schema=_schema(f"{title}Request", request_schema)
                    ),
                    response=foxglove.MessageSchema(
                        encoding="json", schema=_schema(f"{title}Response", _RESPONSE_SCHEMA)
                    ),
                ),
                handler=handler_for(name),
            )
        )
    return services


def live_only_filter(channel: Any) -> bool:
    """MCAP sink filter: keep everything except the recorder's own topics."""
    return not channel.topic.startswith(LIVE_ONLY_PREFIX)


class StatusPublisher:
    """Logs ``/recorder/status`` on a timer, off the control loop's thread."""

    def __init__(self, controls: EpisodeControls, context: foxglove.Context) -> None:
        self._controls = controls
        self._channel = foxglove.Channel(
            TOPIC_STATUS, schema=STATUS_SCHEMA, message_encoding="json", context=context
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="recorder-status", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)
        self.publish()

    def publish(self) -> None:
        self._channel.log(self._controls.snapshot())

    def _run(self) -> None:
        while not self._stop.wait(1.0 / STATUS_HZ):
            self.publish()


def control_capabilities() -> list[Capability]:
    return [Capability.Services]

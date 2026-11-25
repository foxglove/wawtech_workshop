"""Serve a recording session's Foxglove context over a WebSocket server.

Shared by ``record-episode`` and ``rollout-episode``: every message logged on the
session's context reaches both the episode MCAP and any connected Foxglove app.
"""

from __future__ import annotations

from typing import Any

import foxglove

from wawtech_workshop.recorder import urdf

DEFAULT_HINT = (
    "Is another recorder or lerobot-teleoperate using it? Pass --stream_port=<port>, or "
    "--stream=false to record without streaming."
)


def start_stream_server(
    context: foxglove.Context,
    host: str,
    port: int,
    *,
    name: str = "wawtech recorder",
    services: Any = None,
    capabilities: Any = None,
    hint: str = DEFAULT_HINT,
) -> Any:
    """Start the Foxglove server on ``context`` and return it.

    The 3D panel's URDF layer fetches the arm model from the server, so the asset
    handler for ``assets/urdf`` is always attached. ``services`` and
    ``capabilities`` go on the server as given (recorder controls). A port that
    cannot be bound raises ``RuntimeError`` ending in ``hint``.
    """
    urdf.check_assets()
    extra: dict[str, Any] = {"asset_handler": urdf.make_asset_handler()}
    if services is not None:
        extra["services"] = services
    if capabilities is not None:
        extra["capabilities"] = capabilities
    try:
        server = foxglove.start_server(name=name, host=host, port=port, context=context, **extra)
    except RuntimeError as exc:
        raise RuntimeError(f"Could not stream on {host}:{port} ({exc}). {hint}") from exc
    print(f"Streaming live: open ws://{host}:{server.port} in Foxglove.")
    return server

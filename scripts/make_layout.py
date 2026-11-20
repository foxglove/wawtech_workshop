"""Write ``docs/foxglove-layout.json``: the layout for ``record-episode --control=foxglove``.

Import it in the Foxglove app (Layouts menu, Import from file...) after connecting
to ``ws://127.0.0.1:8765``. Regenerate with::

    uv run python scripts/make_layout.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import foxglove.layouts as fl

from wawtech_workshop.mcap_schema import TOPIC_ACTION_JOINT_STATES, TOPIC_JOINT_STATES
from wawtech_workshop.recorder.control import (
    SERVICE_DISCARD,
    SERVICE_FINISH,
    SERVICE_QUIT,
    SERVICE_START,
    TOPIC_STATUS,
)
from wawtech_workshop.recorder.urdf import ASSET_ROOT, BASE_FRAME, PACKAGE, URDF_URI

# The URDF the 3D panel loads in "file" mode (desktop app only): an absolute path on
# this machine, so the layout is only good on the machine that generated it.
URDF_FILE = ASSET_ROOT / PACKAGE / "urdf" / "reBot-DevArm_fixend.urdf"

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")
GRIPPER = "gripper"
COLORS = ("#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#76b7b2")
GREEN, RED, AMBER, GREY = "#59a14f", "#e15759", "#f2c14e", "#888888"

OUT = Path(__file__).resolve().parent.parent / "docs" / "foxglove-layout.json"


def button(service: str, text: str, color: str, payload: dict, tooltip: str) -> fl.SplitItem:
    return fl.SplitItem(
        content=fl.ServiceCallPanel(
            title=text,
            config=fl.ServiceCallConfig(
                service_name=service,
                request_payload=json.dumps(payload),
                button_text=text,
                button_tooltip=tooltip,
                button_color=color,
                editing_mode=False,
                timeout_seconds=5,
            ),
        )
    )


def series(label: str, topic: str, color: str, *, dashed: bool) -> fl.PlotSeries:
    return fl.PlotSeries(
        value=f'{topic}.scalars[:]{{label=="{label}.pos"}}.value',
        label=f"{label} {'action' if dashed else 'state'}",
        color=color,
        line_style="dashed" if dashed else "solid",
        timestamp_method="publishTime",
    )


def joint_plot(title: str, joints: tuple[str, ...], colors: tuple[str, ...]) -> fl.SplitItem:
    paths = []
    for joint, color in zip(joints, colors, strict=True):
        paths.append(series(joint, "/observation/state", color, dashed=False))
        paths.append(series(joint, "/action/state", color, dashed=True))
    return fl.SplitItem(
        content=fl.PlotPanel(
            title=title,
            config=fl.PlotConfig(paths=paths, show_legend=True, min_x_value=0, max_x_value=10),
        )
    )


def camera(name: str) -> fl.SplitItem:
    return fl.SplitItem(
        content=fl.ImagePanel(
            title=name,
            config=fl.ImageConfig(
                image_mode=fl.ImageModeConfig(image_topic=f"/observation/images/{name}")
            ),
        )
    )


GHOST_OPACITY = 0.35
GHOST_COLOR = "#4e79a7"


def urdf_layer(
    source: str,
    instance_id: str,
    label: str,
    joint_states_topic: str,
    *,
    opacity: float = 1,
    fallback_color: str = "#cccccc",
    outlines: bool = True,
) -> dict:
    """A URDF layer driven by the joint states on ``joint_states_topic``."""
    return {
        "layerId": "foxglove.Urdf",
        "instanceId": instance_id,
        "label": label,
        "visible": True,
        "frameLocked": True,
        "sourceType": "filePath" if source == "file" else "url",
        "url": URDF_URI if source == "url" else "",
        "filePath": str(URDF_FILE) if source == "file" else "",
        "controlMode": "jointStates",
        "jointStatesTopic": joint_states_topic,
        "displayMode": "auto",
        "fallbackColor": fallback_color,
        "showAxis": False,
        "axisScale": 1,
        "showOutlines": outlines,
        "opacity": opacity,
        "framePrefix": "",
        "parameter": "",
        "topic": "",
    }


def arm_3d(source: str = "file") -> fl.SplitItem:
    """The arm model: a URDF layer that moves from ``/joint_states`` (control mode
    "Joint states"). ``source="file"`` loads the URDF from disk (desktop app);
    ``"url"`` fetches it from the recorder, which serves it at ``URDF_URI``."""
    config = {
        "cameraState": {
            "distance": 1.1,
            "perspective": True,
            "phi": 65,
            "thetaOffset": 45,
            "targetOffset": [0, 0, 0.15],
            "target": [0, 0, 0],
            "targetOrientation": [0, 0, 0, 1],
            "fovy": 45,
            "near": 0.05,
            "far": 5000,
        },
        "followMode": "follow-none",
        "followTf": BASE_FRAME,
        "scene": {"meshUpAxis": "z_up", "transforms": {"visible": False}},
        "transforms": {},
        "topics": {},
        "layers": {
            "grid": {
                "layerId": "foxglove.Grid",
                "instanceId": "rebot-grid",
                "label": "Grid",
                "visible": True,
                "frameLocked": True,
                "size": 1,
                "divisions": 10,
                "lineWidth": 1,
                "color": "#248eff",
                "position": [0, 0, 0],
                "rotation": [0, 0, 0],
            },
            "urdf": urdf_layer(source, "rebot-urdf", "reBot B601 (follower)", TOPIC_JOINT_STATES),
            "urdf_leader": urdf_layer(
                source,
                "rebot-urdf-leader",
                "Leader (ghost)",
                TOPIC_ACTION_JOINT_STATES,
                opacity=GHOST_OPACITY,
                fallback_color=GHOST_COLOR,
                outlines=False,
            ),
        },
        "publish": {"type": "point"},
        "imageMode": {},
    }
    return fl.SplitItem(content=fl.Panel(panel_type="3D", title="Arm", config=config, version=1))


def build(source: str = "file") -> fl.Layout:
    state = fl.IndicatorPanel(
        title="Recorder",
        config=fl.IndicatorConfig(
            path=f"{TOPIC_STATUS}.state",
            style="background",
            font_size=36,
            rules=[
                fl.IndicatorRule(operator="=", raw_value="recording", color=RED, label="RECORDING"),
                fl.IndicatorRule(operator="=", raw_value="idle", color=GREEN, label="READY"),
                fl.IndicatorRule(operator="=", raw_value="resetting", color=AMBER, label="RESET"),
                fl.IndicatorRule(operator="=", raw_value="stopped", color=GREY, label="STOPPED"),
            ],
            fallback_color=GREY,
            fallback_label="No recorder",
        ),
    )
    details = fl.RawMessagesPanel(
        title="Status",
        config=fl.RawMessagesConfig(topic_path=TOPIC_STATUS, expansion="all"),
    )
    controls = fl.SplitContainer(
        direction="column",
        items=[
            button(SERVICE_START, "Start", GREEN, {}, "Begin the next episode (when ready)"),
            button(SERVICE_FINISH, "Success", GREEN, {"outcome": "success"}, "Save as a success"),
            button(SERVICE_FINISH, "Failure", AMBER, {"outcome": "failure"}, "Save as a failure"),
            button(SERVICE_DISCARD, "Discard", RED, {}, "Throw this episode away"),
            button(SERVICE_QUIT, "Quit", GREY, {}, "End the session; a running episode is saved"),
        ],
    )
    left = fl.SplitContainer(
        direction="column",
        items=[
            fl.SplitItem(proportion=1, content=state),
            fl.SplitItem(proportion=3, content=controls),
            fl.SplitItem(proportion=2, content=details),
        ],
    )
    right = fl.SplitContainer(
        direction="column",
        items=[
            fl.SplitItem(
                proportion=2,
                content=fl.SplitContainer(
                    direction="row", items=[arm_3d(source), camera("wrist"), camera("front")]
                ),
            ),
            fl.SplitItem(
                proportion=2,
                content=fl.SplitContainer(
                    direction="row",
                    items=[
                        joint_plot("Arm: state (solid) vs action (dashed)", JOINTS, COLORS),
                        joint_plot("Gripper: state vs action", (GRIPPER,), (COLORS[0],)),
                    ],
                ),
            ),
        ],
    )
    return fl.Layout(
        content=fl.SplitContainer(
            direction="row",
            items=[
                fl.SplitItem(proportion=1, content=left),
                fl.SplitItem(proportion=4, content=right),
            ],
        )
    )


# SDK panel type -> the type id the app registers (checked against the app's panel list).
APP_PANEL_TYPES = {"ServiceCall": "CallService"}


def to_app_format(layout: fl.Layout) -> dict:
    """The file format the Foxglove app imports and exports.

    The SDK's layout schema (``{"version", "content"}``) is not accepted by the
    app's Import from file ("Incompatible layout"). The app wants ``configById``
    (one config per panel, keyed ``<PanelType>!<id>``) and ``layout``, a binary
    tree of splits that refers to the panels by id.
    """
    configs: dict[str, dict] = {}

    def convert(node: dict) -> str | dict:
        if node["type"] == "panel":
            panel_type = APP_PANEL_TYPES.get(node["panelType"], node["panelType"])
            panel_id = f"{panel_type}!{len(configs) + 1:03x}"
            config = dict(node["config"])
            if node.get("title"):
                config["foxglovePanelTitle"] = node["title"]
            configs[panel_id] = config
            return panel_id
        items = node["items"]
        return chain(node["direction"], [(i["proportion"], convert(i["content"])) for i in items])

    def chain(direction: str, items: list[tuple[float, str | dict]]) -> str | dict:
        if len(items) == 1:
            return items[0][1]
        (first_weight, first), rest = items[0], items[1:]
        total = first_weight + sum(weight for weight, _ in rest)
        return {
            "direction": direction,
            "first": first,
            "second": chain(direction, rest),
            "splitPercentage": round(100 * first_weight / total, 2),
        }

    tree = convert(json.loads(layout.to_json())["content"])
    return {
        "configById": configs,
        "globalVariables": {},
        "userNodes": {},
        "playbackConfig": {"speed": 1},
        "layout": tree,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--urdf-source",
        choices=["file", "url"],
        default="file",
        help="file: read the URDF from this checkout (desktop app, absolute path); "
        "url: fetch it from the running recorder",
    )
    args = parser.parse_args()
    if args.urdf_source == "file" and not URDF_FILE.is_file():
        raise SystemExit(f"URDF not found at {URDF_FILE}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(to_app_format(build(args.urdf_source)), indent=2) + "\n")
    print(f"Wrote {OUT} (URDF source: {args.urdf_source})")


if __name__ == "__main__":
    main()

"""The generated Foxglove layout is in the format the app imports."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _script():
    spec = importlib.util.spec_from_file_location("make_layout", ROOT / "scripts/make_layout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _panel_ids(node) -> list[str]:
    if isinstance(node, str):
        return [node]
    assert node["direction"] in ("row", "column") and 0 < node["splitPercentage"] < 100
    return _panel_ids(node["first"]) + _panel_ids(node["second"])


def test_layout_uses_the_apps_import_format() -> None:
    layout = _script().to_app_format(_script().build())
    assert {"configById", "layout"} <= set(layout) and "content" not in layout
    ids = _panel_ids(layout["layout"])
    assert sorted(ids) == sorted(layout["configById"])
    assert all("!" in panel_id for panel_id in ids)
    services = {
        config["serviceName"]
        for panel_id, config in layout["configById"].items()
        if panel_id.startswith("CallService!")
    }
    assert services == {
        "/recorder/start",
        "/recorder/finish",
        "/recorder/discard",
        "/recorder/quit",
    }


def test_committed_layout_is_current() -> None:
    script = _script()
    committed = json.loads((ROOT / "docs/foxglove-layout.json").read_text())
    fresh = script.to_app_format(script.build())
    # The file-mode path is absolute, so it differs between checkouts.
    for layout in (committed, fresh):
        for key, config in layout["configById"].items():
            if key.startswith("3D!"):
                config["layers"]["urdf"]["filePath"] = "<path>"
    assert committed == fresh, "run scripts/make_layout.py"


def test_3d_panel_moves_the_urdf_from_the_recorders_joint_states() -> None:
    from wawtech_workshop.recorder import urdf

    layout = _script().to_app_format(_script().build())
    [config] = [c for k, c in layout["configById"].items() if k.startswith("3D!")]
    layer = config["layers"]["urdf"]
    assert layer["controlMode"] == "jointStates"
    assert layer["jointStatesTopic"] == "/joint_states"
    assert layer["sourceType"] == "filePath"
    assert Path(layer["filePath"]).is_absolute() and Path(layer["filePath"]).is_file()
    assert config["followTf"] == urdf.BASE_FRAME

    via_url = _script().to_app_format(_script().build("url"))
    [url_config] = [c for k, c in via_url["configById"].items() if k.startswith("3D!")]
    assert url_config["layers"]["urdf"]["sourceType"] == "url"
    assert url_config["layers"]["urdf"]["url"] == urdf.URDF_URI


def test_3d_panel_has_a_translucent_ghost_of_the_leader() -> None:
    layout = _script().to_app_format(_script().build())
    [config] = [c for k, c in layout["configById"].items() if k.startswith("3D!")]
    follower, ghost = config["layers"]["urdf"], config["layers"]["urdf_leader"]
    assert follower["jointStatesTopic"] == "/joint_states" and follower["opacity"] == 1
    assert ghost["jointStatesTopic"] == "/action/joint_states"
    assert 0 < ghost["opacity"] < 1
    assert ghost["instanceId"] != follower["instanceId"]
    assert ghost["filePath"] == follower["filePath"] and ghost["controlMode"] == "jointStates"

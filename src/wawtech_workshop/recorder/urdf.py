"""The reBot B601 model for Foxglove's 3D panel.

``/joint_states`` (``foxglove.JointStates``) carries the follower's joint angles in
radians under the URDF's joint names, and ``/tf`` a fixed ``world`` -> ``base_link``
frame. A URDF layer in the 3D panel set to "Joint states" control mode moves the
arm from them. The recorder serves ``assets/urdf`` to the app as ``package://``
URLs (the app asks the WebSocket server for the URDF and meshes), so the layer's
URL is :data:`URDF_URI`.

LeRobot reports joint positions in degrees, zeroed at the arm's calibration pose,
the same zero the URDF uses, with direction +1 for every joint. The gripper has no
URDF joint and is left out.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

PACKAGE = "rebot-devarm_description_fixend"
URDF_URI = f"package://{PACKAGE}/urdf/reBot-DevArm_fixend.urdf"
ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets" / "urdf"
WORLD_FRAME = "world"
BASE_FRAME = "base_link"

# LeRobot joint -> URDF joint. The model spells the third one "join3".
URDF_JOINTS: dict[str, str] = {
    "shoulder_pan": "joint1",
    "shoulder_lift": "joint2",
    "elbow_flex": "join3",
    "wrist_flex": "joint4",
    "wrist_yaw": "joint5",
    "wrist_roll": "joint6",
}


def urdf_joint_names(state_names: tuple[str, ...]) -> dict[str, str]:
    """State label (``shoulder_pan.pos``) -> URDF joint, for the labels the model has."""
    return {
        label: URDF_JOINTS[label.removesuffix(".pos")]
        for label in state_names
        if label.removesuffix(".pos") in URDF_JOINTS
    }


def joint_positions_rad(
    joints: Mapping[str, str], state: Mapping[str, float]
) -> list[tuple[str, float]]:
    """``(urdf_joint, radians)`` for each mapped label, in label order."""
    return [(urdf, math.radians(float(state[label]))) for label, urdf in joints.items()]


def make_asset_handler(root: Path = ASSET_ROOT):
    """Foxglove ``asset_handler`` for ``package://`` URLs under ``root``.

    Only files inside ``root`` are served; anything else, including ``..``
    traversal, returns ``None``. Every request is logged, so a URDF that will not
    load in the app shows which path it asked for and whether it was found.
    """
    root = root.resolve()

    def handler(uri: str) -> bytes | None:
        prefix = "package://"
        if not uri.startswith(prefix):
            logger.warning("Foxglove asked for %s: only package:// URLs are served.", uri)
            return None
        try:
            path = (root / uri[len(prefix) :]).resolve()
            path.relative_to(root)
            data = path.read_bytes() if path.is_file() else None
        except (ValueError, OSError):
            data = None
        if data is None:
            logger.warning("Foxglove asked for %s: not found under %s.", uri, root)
        else:
            logger.info("Served %s (%d bytes).", uri, len(data))
        return data

    return handler


def check_assets(root: Path = ASSET_ROOT) -> bool:
    """Warn at startup if the model is missing; the 3D panel would then show no arm."""
    urdf_file = root / PACKAGE / "urdf" / "reBot-DevArm_fixend.urdf"
    if urdf_file.is_file():
        return True
    logger.warning(
        "Arm model not found at %s, so Foxglove cannot load the URDF. It is looked up "
        "next to the source tree (assets/urdf in the repo); run record-episode from "
        "an editable checkout.",
        urdf_file,
    )
    return False

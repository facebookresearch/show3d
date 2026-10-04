# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Posed hand meshes of the SHOW3D hand_pose models, in the scene's world frame.

From hand_pose v3 on, every hand-frame with world geometry ships in three hand
models: UmeTrack (the solve, ``hand_pose_umetrack.json``), MANO
(``hand_pose_mano.json``) and MHR (``hand_pose_mhr.json``), each with a
per-subject profile under ``hand_pose/hand_profiles/``; hand_pose v1 and v2 ship
UmeTrack only, as ``hand_pose.json``. Posing a model needs that model's optional
packages, so its code lives in its own module, loaded by name:

* ``hand_mesh_umetrack``: HOT3D's UmeTrack loader (torch, projectaria_tools and
  HOT3D's ``hot3d`` folder on ``PYTHONPATH``).
* ``hand_mesh_mano``: smplx and torch, plus the MANO model files.
* ``hand_mesh_mhr``: the public ``mhr`` package, plus MHR's asset folder.

Each of them defines ``create_mesher(profile_path, scene_payload, asset_dir)``,
returning a :class:`HandMesher`. This module itself needs only the core
dependencies::

    scene = HandMeshScene(root, "ISH822", "aria_inspecting_3ab0", "mano",
                          asset_dir="mano_v1_2/models")
    mesh = scene.mesh(frame_index=120, slot=RIGHT_SLOT)  # HandMesh or None
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast, Protocol

import numpy as np
from numpy.typing import NDArray

from .dataset import ACCEPT_CONFIDENCE_THRESHOLD, FloatArray, LEGACY_HAND_POSE_VERSIONS

HAND_MODELS: tuple[str, ...] = ("umetrack", "mano", "mhr")
# MANO and MHR ship from hand_pose v3 on.
DEFAULT_MESH_HAND_POSE_VERSION: str = "v3"
LEFT_SLOT: int = 0
RIGHT_SLOT: int = 1


@dataclass(frozen=True)
class HandMesh:
    """One posed hand in the scene's world frame, in millimeters.

    ``landmarks_world_mm`` follows the UmeTrack landmark order of
    ``hand_pose_umetrack.json`` (21 for UmeTrack, the first 20 for MANO); MHR
    carries none.
    """

    vertices_world_mm: FloatArray
    faces: NDArray[np.int64]
    landmarks_world_mm: FloatArray | None


class HandMesher(Protocol):
    """Poses one hand model; each ``hand_mesh_<model>`` module builds one."""

    def posed_mesh(self, hand: Mapping[str, object], slot: int) -> HandMesh | None:
        """The mesh of one hand entry of a scene file, or None when the entry
        has no world geometry."""
        ...


def _scene_file_name(version: str, model: str) -> str:
    if version not in LEGACY_HAND_POSE_VERSIONS:
        return f"hand_pose_{model}.json"
    if model != "umetrack":
        raise ValueError(f"{model} hand poses ship from hand_pose v3 on, not {version}")
    if version == "v1":
        # v2 re-tracked three v1 scenes with the hand profile that ships today.
        raise ValueError("UmeTrack hand meshes need hand_pose v2 or later, not v1")
    return "hand_pose.json"


class HandMeshScene:
    """One scene's posed hand meshes in one hand model."""

    def __init__(
        self,
        root: str | Path,
        subject: str,
        scene: str,
        model: str,
        *,
        asset_dir: str | Path | None = None,
        version: str = DEFAULT_MESH_HAND_POSE_VERSION,
        confidence_threshold: float = ACCEPT_CONFIDENCE_THRESHOLD,
    ) -> None:
        if model not in HAND_MODELS:
            raise ValueError(f"model must be one of {HAND_MODELS}, got {model!r}")
        hand_pose_dir = Path(root) / "hand_pose"
        scene_path = (
            hand_pose_dir
            / version
            / "scenes"
            / subject
            / scene
            / _scene_file_name(version, model)
        )
        with scene_path.open() as f:
            payload = cast(dict[str, object], json.load(f))
        # The UmeTrack file maps frame keys to frames; MANO and MHR files nest
        # them under "frames" beside their own fields.
        frames = payload.get("frames", payload)
        self.frames: Mapping[str, object] = cast(Mapping[str, object], frames)
        self.confidence_threshold: float = confidence_threshold
        module = importlib.import_module(f".hand_mesh_{model}", __package__)
        # One profile per subject serves hand_pose v2 and later.
        self._mesher: HandMesher = module.create_mesher(
            hand_pose_dir / "hand_profiles" / subject / f"profile_{model}.json",
            payload,
            None if asset_dir is None else Path(asset_dir),
        )

    def hand(self, frame_index: int, slot: int) -> Mapping[str, object] | None:
        """The scene file's entry of one hand, or None when the hand is absent or
        at or below ``confidence_threshold``."""
        frame = self.frames.get(str(frame_index))
        if not isinstance(frame, dict):
            return None
        hands = frame.get("hand_poses")
        hand = hands.get(str(slot)) if isinstance(hands, dict) else None
        if not isinstance(hand, dict):
            return None
        confidence = hand.get("confidence")
        if not isinstance(confidence, (int, float)):
            return None
        if confidence <= self.confidence_threshold:
            return None
        return cast(Mapping[str, object], hand)

    def mesh(self, frame_index: int, slot: int) -> HandMesh | None:
        hand = self.hand(frame_index, slot)
        return None if hand is None else self._mesher.posed_mesh(hand, slot)

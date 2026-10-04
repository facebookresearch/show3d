# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""UmeTrack hand meshes through HOT3D's public UmeTrack loader.

Needs torch and HOT3D's ``hot3d`` folder on ``PYTHONPATH`` (its
``data_loaders`` package needs projectaria_tools).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from data_loaders.UmeTrackHandDataProvider import (
    load_hand_model_from_file,
    skin_landmarks,
    skin_vertices,
)
from numpy.typing import NDArray

from .hand_mesh import HandMesh, RIGHT_SLOT


class UmeTrackMesher:
    def __init__(self, profile_path: Path) -> None:
        model = load_hand_model_from_file(str(profile_path))
        if model is None or model.mesh_triangles is None:
            raise ValueError(f"{profile_path} has no UmeTrack hand mesh")
        self._model = model
        self._faces: NDArray[np.int64] = model.mesh_triangles.numpy().astype(np.int64)
        # Mirroring the left-hand model into a right hand reverses the winding.
        self._right_faces: NDArray[np.int64] = self._faces[:, ::-1].copy()

    def posed_mesh(self, hand: Mapping[str, object], slot: int) -> HandMesh | None:
        rotation = hand.get("wrist_rotation")
        translation_mm = hand.get("wrist_translation")
        joint_angles = hand.get("joint_angles")
        if rotation is None or translation_mm is None or joint_angles is None:
            return None
        wrist_m = np.eye(4)
        wrist_m[:3, :3] = np.asarray(rotation, dtype=np.float64)
        wrist_m[:3, 3] = np.asarray(translation_mm, dtype=np.float64) / 1000.0
        # The profile describes a left hand; HOT3D poses a right hand by
        # flipping the wrist frame's x axis.
        if slot == RIGHT_SLOT:
            wrist_m[:, 0] *= -1.0
        wrist = torch.from_numpy(wrist_m)
        angles = torch.tensor(joint_angles, dtype=torch.float64)
        vertices_m = skin_vertices(self._model, angles, wrist)
        landmarks_m = skin_landmarks(self._model, angles, wrist)
        return HandMesh(
            vertices_world_mm=vertices_m.numpy() * 1000.0,
            faces=self._right_faces if slot == RIGHT_SLOT else self._faces,
            landmarks_world_mm=landmarks_m.numpy() * 1000.0,
        )


def create_mesher(
    profile_path: Path, scene_payload: Mapping[str, object], asset_dir: Path | None
) -> UmeTrackMesher:
    return UmeTrackMesher(profile_path)

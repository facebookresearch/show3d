# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""MANO hand meshes through smplx.

Needs torch, smplx and a folder holding ``MANO_LEFT.pkl`` and ``MANO_RIGHT.pkl``
from https://mano.is.tue.mpg.de, with their chumpy objects removed (smplx's
``tools/clean_ch.py``) so they load without chumpy installed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import smplx
import torch
from numpy.typing import NDArray

from .hand_mesh import HandMesh

# HOT3D's MANO landmarks: the 16 smplx joints, then these fingertip vertices
# (thumb, index, middle, ring, pinky), reordered into UmeTrack's first 20
# landmarks.
MANO_FINGERTIP_VERTICES: list[int] = [744, 320, 443, 554, 671]
MANO_TO_UMETRACK_LANDMARKS: list[int] = [
    16, 17, 18, 19, 20, 0, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9,
]  # fmt: skip


class ManoMesher:
    def __init__(self, profile_path: Path, model_dir: Path) -> None:
        with profile_path.open() as f:
            profile = json.load(f)
        num_betas = int(profile["num_shape_coeffs"])
        self._betas: torch.Tensor = torch.tensor(
            [profile["betas"]], dtype=torch.float64
        )
        # Poses are offsets from the MANO mean hand, over all 45 joint angles.
        self._layers: list[smplx.MANO] = [
            smplx.MANO(
                str(model_dir),
                is_rhand=is_rhand,
                use_pca=False,
                flat_hand_mean=False,
                num_betas=num_betas,
                dtype=torch.float64,
            )
            for is_rhand in (False, True)
        ]
        left, right = (cast(torch.Tensor, layer.shapedirs) for layer in self._layers)
        # The fix HOT3D applies to the left model's first shape direction
        # (https://github.com/vchoutas/smplx/issues/48).
        if torch.sum(torch.abs(left[:, 0, :] - right[:, 0, :])) < 1:
            left[:, 0, :] *= -1
        self._faces: list[NDArray[np.int64]] = [
            np.asarray(layer.faces, dtype=np.int64) for layer in self._layers
        ]

    def posed_mesh(self, hand: Mapping[str, object], slot: int) -> HandMesh:
        # global_transform is the root's axis-angle, then a translation in meters.
        transform = torch.tensor([hand["global_transform"]], dtype=torch.float64)
        with torch.no_grad():
            output = self._layers[slot](
                betas=self._betas,
                global_orient=transform[:, :3],
                hand_pose=torch.tensor([hand["pose_parameters"]], dtype=torch.float64),
                transl=transform[:, 3:],
            )
        vertices = output.vertices[0]
        landmarks = torch.cat([output.joints[0], vertices[MANO_FINGERTIP_VERTICES]])
        return HandMesh(
            vertices_world_mm=vertices.numpy() * 1000.0,
            faces=self._faces[slot],
            landmarks_world_mm=landmarks[MANO_TO_UMETRACK_LANDMARKS].numpy() * 1000.0,
        )


def create_mesher(
    profile_path: Path, scene_payload: Mapping[str, object], asset_dir: Path | None
) -> ManoMesher:
    if asset_dir is None:
        raise ValueError(
            "MANO needs the folder holding MANO_LEFT.pkl and MANO_RIGHT.pkl"
        )
    return ManoMesher(profile_path, asset_dir)

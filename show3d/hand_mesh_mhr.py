# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""MHR hand meshes through the public ``mhr`` package.

Needs torch, ``mhr``, pymomentum and MHR's asset folder, from
https://github.com/facebookresearch/MHR.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pymomentum.quaternion as pq
import torch
from mhr.mhr import MHR
from numpy.typing import NDArray

from .dataset import FloatArray
from .hand_mesh import HandMesh

SIDE_PREFIX: tuple[str, str] = ("l", "r")
SIDE_NAME: tuple[str, str] = ("left", "right")
CM_PER_M = 100.0
MM_PER_CM = 10.0


def _descendants(joint: int, parents: Sequence[int]) -> set[int]:
    """``joint`` and every joint below it."""
    found = {joint}
    for candidate in range(len(parents)):
        chain = candidate
        while 0 <= chain < len(parents) and chain not in found:
            parent = parents[chain]
            chain = parent if parent != chain else -1
        if chain in found:
            found.add(candidate)
    return found


class MhrMesher:
    def __init__(
        self,
        profile_path: Path,
        finger_parameter_names: Mapping[str, Sequence[str]],
        asset_dir: Path,
    ) -> None:
        with profile_path.open() as f:
            profile = json.load(f)
        # The fit used no pose correctives, so the mesh leaves them out too.
        self._model: MHR = MHR.from_files(
            folder=asset_dir,
            device=torch.device("cpu"),
            lod=1,
            wants_pose_correctives=False,
        )
        character = self._model.character
        names = list(character.parameter_transform.names)
        # forward() takes the model parameters without the identity and
        # expression coefficients that end the parameter list.
        num_parameters = (
            len(names)
            - self._model.get_num_identity_blendshapes()
            - self._model.get_num_face_expression_blendshapes()
        )
        self._identity: torch.Tensor = torch.tensor(
            [profile["identity_coefficients"]], dtype=torch.float32
        )
        joint_names = list(character.skeleton.joint_names)
        parents = [int(p) for p in character.skeleton.joint_parents]
        weights = np.asarray(character.skin_weights.weight)
        indices = np.asarray(character.skin_weights.index)
        dominant = indices[np.arange(len(indices)), weights.argmax(1)]
        all_faces = np.asarray(character.mesh.faces, dtype=np.int64)
        self._finger_columns: list[list[int]] = []
        self._parameters: list[torch.Tensor] = []
        self._wrist_joint: list[int] = []
        self._vertices: list[NDArray[np.int64]] = []
        self._faces: list[NDArray[np.int64]] = []
        for slot, prefix in enumerate(SIDE_PREFIX):
            self._finger_columns.append(
                [names.index(name) for name in finger_parameter_names[str(slot)]]
            )
            parameters = torch.zeros(num_parameters)
            for name, value in profile["hand_scales"][SIDE_NAME[slot]].items():
                parameters[names.index(name)] = value
            self._parameters.append(parameters)
            wrist = joint_names.index(f"{prefix}_wrist")
            self._wrist_joint.append(wrist)
            # The hand: the vertices skinned mostly to the wrist or a joint below.
            keep = np.nonzero(np.isin(dominant, sorted(_descendants(wrist, parents))))[
                0
            ]
            remap = np.full(len(dominant), -1, dtype=np.int64)
            remap[keep] = np.arange(len(keep))
            self._vertices.append(keep)
            self._faces.append(remap[all_faces[np.isin(all_faces, keep).all(1)]])

    def posed_mesh(self, hand: Mapping[str, object], slot: int) -> HandMesh:
        vertices_cm, skeleton_state = self._posed(hand, slot)
        return HandMesh(
            vertices_world_mm=self._to_world_mm(
                vertices_cm[self._vertices[slot]], skeleton_state, hand, slot
            ),
            faces=self._faces[slot],
            landmarks_world_mm=None,
        )

    def _posed(
        self, hand: Mapping[str, object], slot: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MHR's vertices and skeleton state in centimeters, in MHR's frame, with
        the body at zero and this hand's finger parameters and scales set."""
        parameters = self._parameters[slot].clone()
        parameters[self._finger_columns[slot]] = torch.tensor(
            cast(list[float], hand["finger_parameters"]), dtype=torch.float32
        )
        with torch.no_grad():
            vertices, skeleton_state = self._model.forward(
                self._identity, parameters[None], None
            )
        return vertices[0], skeleton_state[0]

    def _to_world_mm(
        self,
        points_cm: torch.Tensor,
        skeleton_state: torch.Tensor,
        hand: Mapping[str, object],
        slot: int,
    ) -> FloatArray:
        """Move points rigidly from MHR's wrist joint (``R_0``, ``t_0``) to the
        world wrist (``R_w``, ``t_w``): ``R_w R_0^T (p - t_0) + t_w``."""
        # A skeleton state row is translation, quaternion (x, y, z, w), scale.
        wrist = skeleton_state[self._wrist_joint[slot]]
        rotation = pq.multiply(
            pq.from_axis_angle(
                torch.tensor(hand["wrist_rotation"], dtype=torch.float32)
            ),
            pq.inverse(wrist[3:7]),
        )
        translation_cm = CM_PER_M * torch.tensor(
            hand["wrist_translation"], dtype=torch.float32
        )
        world_cm = (
            pq.rotate_vector(rotation.expand(len(points_cm), 4), points_cm - wrist[:3])
            + translation_cm
        )
        return world_cm.numpy().astype(np.float64) * MM_PER_CM


def create_mesher(
    profile_path: Path, scene_payload: Mapping[str, object], asset_dir: Path | None
) -> MhrMesher:
    if asset_dir is None:
        raise ValueError("MHR needs its asset folder")
    names = cast(Mapping[str, Sequence[str]], scene_payload["finger_parameter_names"])
    return MhrMesher(profile_path, names, asset_dir)

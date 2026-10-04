# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pinhole camera projection for SHOW3D egocentric views.

The released headset videos are fisheye-undistorted, so a plain pinhole applies.
Calibration ships ``T_world_from_camera`` (the 4x4 camera-to-world transform) and
pinhole intrinsics ``fx, fy, cx, cy``. World-to-camera is the inverse:

    Xc = R^T (Xw - t)                       # R, t from T_world_from_camera
    u = fx * Xc / Zc + cx,  v = fy * Yc / Zc + cy

Points behind the camera (``Zc <= 0``) or outside the image are marked invalid.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .dataset import CameraCalibration, FloatArray


def world_to_camera(
    points_world_mm: FloatArray, t_world_from_camera: FloatArray
) -> FloatArray:
    """Transform world-mm points into the camera frame.

    ``T_world_from_camera`` is camera-to-world, so this applies its inverse,
    ``Xc = R^T (Xw - t)`` (equivalently ``(Xw - t) @ R`` for row-vector batches).
    """
    rotation = t_world_from_camera[:3, :3]
    translation = t_world_from_camera[:3, 3]
    return (points_world_mm - translation) @ rotation


def project_to_image(
    points_world_mm: FloatArray, calibration: CameraCalibration
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Project world-mm points to pixels; return ``(uv, valid)`` (both length N).

    Raises if the frame has no valid ``t_world_from_camera``.
    """
    t_world_from_camera = calibration.t_world_from_camera
    if t_world_from_camera is None:
        raise ValueError("frame has no valid t_world_from_camera")
    cam = world_to_camera(points_world_mm, t_world_from_camera)
    z = cam[:, 2]
    safe_z = np.where(z > 0.0, z, np.nan)
    u = calibration.fx * cam[:, 0] / safe_z + calibration.cx
    v = calibration.fy * cam[:, 1] / safe_z + calibration.cy
    uv = np.stack([u, v], axis=1)
    valid = (
        (z > 0.0)
        & (u >= 0.0)
        & (u < calibration.image_width)
        & (v >= 0.0)
        & (v < calibration.image_height)
    )
    return uv, valid

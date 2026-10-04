# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np

from ..camera import project_to_image, world_to_camera
from ..dataset import CameraCalibration, FloatArray


def _calibration(t_world_from_camera: FloatArray | None) -> CameraCalibration:
    return CameraCalibration(
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        image_width=100,
        image_height=100,
        t_world_from_camera=t_world_from_camera,
        is_synthesized=False,
    )


class CameraTest(unittest.TestCase):
    def test_identity_projection_and_validity(self) -> None:
        points = np.asarray([[0.0, 0.0, 1000.0], [10.0, 0.0, 1000.0], [0.0, 0.0, -5.0]])
        uv, valid = project_to_image(points, _calibration(np.eye(4)))
        # (0,0,1000) -> principal point; (10,0,1000) -> u = 100*10/1000 + 50 = 51.
        np.testing.assert_allclose(uv[0], [50.0, 50.0])
        np.testing.assert_allclose(uv[1], [51.0, 50.0])
        self.assertTrue(bool(valid[0]))
        self.assertTrue(bool(valid[1]))
        self.assertFalse(bool(valid[2]))  # behind the camera

    def test_extrinsic_inverse(self) -> None:
        # Camera sits at world (0, 0, -1000), looking +Z. The world origin is then
        # 1000 mm in front of it -> projects to the principal point.
        transform = np.eye(4)
        transform[:3, 3] = [0.0, 0.0, -1000.0]
        cam = world_to_camera(np.asarray([[0.0, 0.0, 0.0]]), transform)
        np.testing.assert_allclose(cam[0], [0.0, 0.0, 1000.0])
        uv, valid = project_to_image(
            np.asarray([[0.0, 0.0, 0.0]]), _calibration(transform)
        )
        np.testing.assert_allclose(uv[0], [50.0, 50.0])
        self.assertTrue(bool(valid[0]))

    def test_out_of_bounds_invalid(self) -> None:
        # u = 100 * 100/1000 + 50 = 60 (in), but a big x lands outside the 100px image.
        points = np.asarray([[1000.0, 0.0, 1000.0]])
        _uv, valid = project_to_image(points, _calibration(np.eye(4)))
        self.assertFalse(bool(valid[0]))

    def test_missing_extrinsic_raises(self) -> None:
        with self.assertRaises(ValueError):
            project_to_image(np.zeros((1, 3)), _calibration(None))


if __name__ == "__main__":
    unittest.main()

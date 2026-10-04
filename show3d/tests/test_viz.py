# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ..dataset import CameraCalibration
from ..hand_mesh import HandMesh
from ..synthetic import build_synthetic_scene

# The visualization library needs matplotlib; the rest of the package does not,
# so guard the import and skip these tests when it is unavailable.
try:
    from ..viz import draw_hand_meshes, run_visualization
except ModuleNotFoundError:
    draw_hand_meshes = None
    run_visualization = None


@unittest.skipUnless(
    run_visualization is not None, "matplotlib is required for the visualization demo"
)
class VizTest(unittest.TestCase):
    def _render(self, mode: str) -> None:
        assert run_visualization is not None
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest_path = build_synthetic_scene(root, num_frames=4)
            out_path = root / f"{mode}.png"
            returned = run_visualization(root, manifest_path, out_path, mode=mode)
            self.assertEqual(returned, out_path)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_geometry_mode(self) -> None:
        self._render("geometry")

    def test_overlay_mode(self) -> None:
        self._render("overlay")

    def test_hand_mesh_nearer_triangle_occludes(self) -> None:
        assert draw_hand_meshes is not None
        # A camera turned 180 degrees about x and moved, so ordering by world z
        # paints the triangles in the opposite order to camera depth.
        t_world_from_camera = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 1000.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        calibration = CameraCalibration(
            fx=100.0,
            fy=100.0,
            cx=50.0,
            cy=50.0,
            image_width=100,
            image_height=100,
            t_world_from_camera=t_world_from_camera,
            is_synthesized=False,
            is_pose_valid=True,
        )

        def facing_triangle(depth_mm: float, half_width_px: float) -> HandMesh:
            # Centered on the optical axis, so it is drawn at full shade.
            corners = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 2.0, 0.0]])
            camera_points = corners * half_width_px * depth_mm / 100.0
            camera_points[:, 2] = depth_mm
            return HandMesh(
                vertices_world_mm=camera_points @ t_world_from_camera[:3, :3].T
                + t_world_from_camera[:3, 3],
                faces=np.array([[0, 1, 2]]),
                landmarks_world_mm=None,
            )

        near, far = (0, 0, 200), (200, 0, 0)
        image = draw_hand_meshes(
            np.zeros((100, 100, 3), dtype=np.uint8),
            [(facing_triangle(300.0, 30.0), near), (facing_triangle(600.0, 45.0), far)],
            calibration,
            alpha=1.0,
        )
        # Both triangles cover the center; only the far one reaches row 10.
        self.assertEqual(tuple(image[50, 50]), near)
        self.assertEqual(tuple(image[10, 50]), far)


if __name__ == "__main__":
    unittest.main()

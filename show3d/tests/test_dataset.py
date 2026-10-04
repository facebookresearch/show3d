# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ..dataset import (
    DEFAULT_VIDEO_FPS,
    load_camera_calibration,
    ObjectPoseFrame,
    sampled_frame_indices,
    Show3DDataset,
    Show3DFrameRef,
    write_frame_manifest_jsonl,
)


class Show3DDatasetTest(unittest.TestCase):
    def test_sampling_uses_source_frame_stride_for_integer_rates(self) -> None:
        self.assertEqual(
            sampled_frame_indices(13, sampling_fps=10.0, video_fps=DEFAULT_VIDEO_FPS),
            [0, 6, 12],
        )
        self.assertEqual(
            sampled_frame_indices(13, sampling_fps=5.0, video_fps=DEFAULT_VIDEO_FPS),
            [0, 12],
        )

    def test_generic_frame_loader_reads_paths_and_pose_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            frame = Show3DFrameRef(
                subject_id="S001",
                scene_id="mug_grab_a1b2",
                frame_index=0,
                object_alias="mug",
            )
            manifest_path = root / "frames.jsonl"
            write_frame_manifest_jsonl(manifest_path, [frame])
            self._write_scene_files(root, frame)

            dataset = Show3DDataset.from_manifest_jsonl(root, manifest_path)
            item = dataset[0]

            self.assertEqual(item.frame, frame)
            self.assertEqual(set(item.views), {"headset0", "headset1"})
            self.assertEqual(item.views["headset0"].video_path.name, "headset0.mp4")
            self.assertEqual(item.views["headset1"].video_path.name, "headset1.mp4")
            self.assertIsNone(item.views["headset0"].image)
            calibration = item.views["headset0"].calibration
            assert calibration is not None
            self.assertEqual(calibration.fx, 450.0)
            self.assertEqual(calibration.image_width, 1024)
            np.testing.assert_allclose(calibration.t_world_from_camera, np.eye(4))
            self.assertEqual(calibration.pose_source, "legacy_unspecified")
            self.assertTrue(calibration.is_pose_valid)
            self.assertIsNone(calibration.pose_contract_version)
            self.assertIsNotNone(item.views["headset1"].calibration)
            self.assertTrue(item.headset_tracking_valid)
            self.assertTrue(item.headset_pose_valid)
            self.assertEqual(item.frame_info_path.name, "frame_info.json")
            self.assertIsNotNone(item.object_pose)
            self.assertIsNotNone(item.left_hand)
            self.assertIsNotNone(item.right_hand)
            assert item.object_pose is not None
            assert item.left_hand is not None
            assert item.right_hand is not None
            self.assertEqual(item.object_pose.confidence, 0.99)
            np.testing.assert_allclose(item.object_pose.rotation, np.eye(3))
            np.testing.assert_allclose(item.object_pose.translation_mm, np.zeros(3))
            self.assertEqual(item.left_hand.confidence, 0.95)
            self.assertEqual(item.right_hand.confidence, 0.0)

    def test_rig_calibration_has_a_static_pose_on_every_frame(self) -> None:
        transform = [
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rig0.json"
            path.write_text(
                json.dumps(
                    {
                        "ImageSizeX": 1280,
                        "ImageSizeY": 1024,
                        "fx": 450.0,
                        "fy": 450.0,
                        "cx": 640.0,
                        "cy": 512.0,
                        "DistortionModel": "PinholePlane",
                        "T_WorldFromCamera": transform,
                        "inverse": {},
                    }
                )
            )
            for frame_index in (0, 1234):
                calibration = load_camera_calibration(path, frame_index)
                assert calibration is not None
                np.testing.assert_allclose(calibration.t_world_from_camera, transform)
                self.assertTrue(calibration.is_pose_valid)
                self.assertEqual(calibration.pose_source, "static")

    def test_pose_vertices_applies_world_from_object_transform(self) -> None:
        # 90deg about +z (world-from-object) then translate by (100, 0, 0) mm.
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        pose = ObjectPoseFrame(
            confidence=1.0,
            rotation=rotation,
            translation_mm=np.asarray([100.0, 0.0, 0.0]),
        )
        world = pose.pose_vertices(np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))
        np.testing.assert_allclose(
            world, np.asarray([[100.0, 1.0, 0.0], [98.0, 0.0, 0.0]])
        )

    def test_single_view_selection_populates_one_view(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            frame = Show3DFrameRef(
                subject_id="S001",
                scene_id="mug_grab_a1b2",
                frame_index=0,
                object_alias="mug",
            )
            manifest_path = root / "frames.jsonl"
            write_frame_manifest_jsonl(manifest_path, [frame])
            self._write_scene_files(root, frame)

            dataset = Show3DDataset.from_manifest_jsonl(
                root, manifest_path, multiview=False
            )
            self.assertEqual(set(dataset[0].views), {"headset0"})

    def test_synthesized_frame_withholds_pose_and_flags_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            frame = Show3DFrameRef(
                subject_id="S001",
                scene_id="mug_grab_a1b2",
                frame_index=0,
                object_alias="mug",
            )
            manifest_path = root / "frames.jsonl"
            write_frame_manifest_jsonl(manifest_path, [frame])
            self._write_scene_files(root, frame, is_synthesized=True)

            item = Show3DDataset.from_manifest_jsonl(root, manifest_path)[0]

            self.assertFalse(item.headset_tracking_valid)
            calibration = item.views["headset0"].calibration
            assert calibration is not None
            self.assertTrue(calibration.is_synthesized)
            self.assertEqual(calibration.pose_source, "legacy_interpolation")
            self.assertFalse(calibration.is_pose_valid)
            self.assertIsNone(calibration.t_world_from_camera)
            self.assertFalse(item.headset_pose_valid)

    def test_explicit_pose_contract_uses_validity_gate(self) -> None:
        scenarios = (
            ("mocap", False, True),
            ("mocap", True, True),
            ("mocap", False, False),
        )
        for pose_source, is_synthesized, is_pose_valid in scenarios:
            with (
                self.subTest(
                    pose_source=pose_source,
                    is_synthesized=is_synthesized,
                    is_pose_valid=is_pose_valid,
                ),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                frame = Show3DFrameRef(
                    subject_id="S001",
                    scene_id="mug_grab_a1b2",
                    frame_index=0,
                    object_alias="mug",
                )
                manifest_path = root / "frames.jsonl"
                write_frame_manifest_jsonl(manifest_path, [frame])
                self._write_scene_files(
                    root,
                    frame,
                    is_synthesized=is_synthesized,
                    pose_source=pose_source,
                    is_pose_valid=is_pose_valid,
                    pose_contract_version=1,
                )

                item = Show3DDataset.from_manifest_jsonl(root, manifest_path)[0]

                calibration = item.views["headset0"].calibration
                assert calibration is not None
                self.assertEqual(item.headset_tracking_valid, not is_synthesized)
                self.assertEqual(item.headset_pose_valid, is_pose_valid)
                self.assertEqual(calibration.pose_source, pose_source)
                self.assertEqual(calibration.is_pose_valid, is_pose_valid)
                self.assertEqual(calibration.pose_contract_version, 1)
                if is_pose_valid:
                    np.testing.assert_allclose(
                        calibration.t_world_from_camera, np.eye(4)
                    )
                else:
                    self.assertIsNone(calibration.t_world_from_camera)

    def test_missing_secondary_calibration_keeps_headset_pose_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = Show3DFrameRef(
                subject_id="S001",
                scene_id="mug_grab_a1b2",
                frame_index=0,
                object_alias="mug",
            )
            manifest_path = root / "frames.jsonl"
            write_frame_manifest_jsonl(manifest_path, [frame])
            self._write_scene_files(root, frame)
            (
                root
                / "scenes"
                / frame.subject_id
                / frame.scene_id
                / "camera_calibration"
                / "headset1.json"
            ).unlink()

            item = Show3DDataset.from_manifest_jsonl(root, manifest_path)[0]

            self.assertTrue(item.headset_tracking_valid)
            self.assertTrue(item.headset_pose_valid)

    def test_pose_contract_rejects_malformed_contracts(self) -> None:
        scenarios = (
            (2, "mocap", True, "Unsupported calibration pose_contract_version"),
            (None, "mocap", True, "require pose_contract_version 1"),
        )
        for version, pose_source, is_pose_valid, error_pattern in scenarios:
            with (
                self.subTest(version=version, pose_source=pose_source),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                frame = Show3DFrameRef(
                    subject_id="S001",
                    scene_id="mug_grab_a1b2",
                    frame_index=0,
                    object_alias="mug",
                )
                manifest_path = root / "frames.jsonl"
                write_frame_manifest_jsonl(manifest_path, [frame])
                self._write_scene_files(
                    root,
                    frame,
                    pose_source=pose_source,
                    is_pose_valid=is_pose_valid,
                    pose_contract_version=version,
                )

                with self.assertRaisesRegex(ValueError, error_pattern):
                    Show3DDataset.from_manifest_jsonl(root, manifest_path)[0]

    def test_valid_headset_pose_can_have_missing_camera_transform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = Show3DFrameRef(
                subject_id="S001",
                scene_id="mug_grab_a1b2",
                frame_index=0,
                object_alias="mug",
            )
            manifest_path = root / "frames.jsonl"
            write_frame_manifest_jsonl(manifest_path, [frame])
            self._write_scene_files(
                root,
                frame,
                pose_source="mocap",
                is_pose_valid=True,
                pose_contract_version=1,
            )
            calibration_path = (
                root
                / "scenes"
                / frame.subject_id
                / frame.scene_id
                / "camera_calibration"
                / "headset0.json"
            )
            payload = json.loads(calibration_path.read_text())
            payload["T_WorldFromCamera_by_index"]["0"]["T_WorldFromCamera"] = None
            calibration_path.write_text(json.dumps(payload))

            item = Show3DDataset.from_manifest_jsonl(root, manifest_path)[0]

            calibration = item.views["headset0"].calibration
            assert calibration is not None
            self.assertTrue(calibration.is_pose_valid)
            self.assertIsNone(calibration.t_world_from_camera)
            self.assertTrue(item.headset_pose_valid)

    def test_unversioned_calibration_keeps_the_legacy_reading(self) -> None:
        # Released files predate the contract: a kept frame parses its matrix
        # permissively, and a synthesized frame skips it.
        scenarios: dict[str, tuple[int, object, bool]] = {
            "kept": (0, [[1.0, 0.0], [0.0, 1.0]], True),
            "synthesized": (1, {"malformed": "matrix"}, False),
        }
        for name, (is_synthesized, transform, is_pose_valid) in scenarios.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                frame = Show3DFrameRef(
                    subject_id="S001",
                    scene_id="mug_grab_a1b2",
                    frame_index=0,
                    object_alias="mug",
                )
                manifest_path = root / "frames.jsonl"
                write_frame_manifest_jsonl(manifest_path, [frame])
                self._write_scene_files(
                    root,
                    frame,
                    is_synthesized=is_synthesized,
                    transform=transform,
                )

                item = Show3DDataset.from_manifest_jsonl(root, manifest_path)[0]

                calibration = item.views["headset0"].calibration
                assert calibration is not None
                self.assertEqual(calibration.is_synthesized, bool(is_synthesized))
                self.assertEqual(calibration.is_pose_valid, is_pose_valid)
                if is_pose_valid:
                    np.testing.assert_allclose(
                        calibration.t_world_from_camera, transform
                    )
                else:
                    self.assertIsNone(calibration.t_world_from_camera)

    def _write_scene_files(
        self,
        root: Path,
        frame: Show3DFrameRef,
        *,
        is_synthesized: object = False,
        pose_source: str | None = None,
        is_pose_valid: bool | None = None,
        pose_contract_version: int | None = None,
        transform: object | None = None,
    ) -> None:
        scene_dir = root / "scenes" / frame.subject_id / frame.scene_id
        scene_dir.mkdir(parents=True)
        (scene_dir / "headset0.mp4").write_bytes(b"")
        (scene_dir / "headset1.mp4").write_bytes(b"")

        calibration_dir = scene_dir / "camera_calibration"
        calibration_dir.mkdir()
        identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        entry: dict[str, object] = {
            "index": 0,
            "T_WorldFromCamera": identity if transform is None else transform,
            "is_synthesized": is_synthesized,
        }
        if pose_source is not None:
            entry["pose_source"] = pose_source
        if is_pose_valid is not None:
            entry["is_pose_valid"] = is_pose_valid
        for name in ("headset0", "headset1"):
            with (calibration_dir / f"{name}.json").open("w") as f:
                payload: dict[str, object] = {
                    "ImageSizeX": 1024,
                    "ImageSizeY": 1280,
                    "fx": 450.0,
                    "fy": 450.0,
                    "cx": 512.0,
                    "cy": 640.0,
                    "DistortionModel": "PinholePlane",
                    "T_WorldFromCamera_by_index": {"0": entry},
                }
                if pose_contract_version is not None:
                    payload["pose_contract_version"] = pose_contract_version
                json.dump(
                    payload,
                    f,
                )

        metadata_dir = scene_dir / "metadata"
        metadata_dir.mkdir()
        (metadata_dir / "frame_info.json").write_text("{}")

        object_pose_dir = (
            root / "object_pose" / "v1" / "scenes" / frame.subject_id / frame.scene_id
        )
        object_pose_dir.mkdir(parents=True)
        with (object_pose_dir / "object_pose.json").open("w") as f:
            json.dump(
                {
                    "0": {
                        "confidence": 0.99,
                        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        "t": [[0.0], [0.0], [0.0]],
                    }
                },
                f,
            )

        hand_pose_dir = (
            root / "hand_pose" / "v2" / "scenes" / frame.subject_id / frame.scene_id
        )
        hand_pose_dir.mkdir(parents=True)
        with (hand_pose_dir / "hand_pose.json").open("w") as f:
            json.dump(
                {
                    "0": {
                        "hand_poses": {
                            "0": {
                                "confidence": 0.95,
                                "landmarks_3d_mm": [[1.0, 0.0, 0.0]],
                            },
                            "1": {
                                "confidence": 0.0,
                                "landmarks_3d_mm": None,
                            },
                        }
                    }
                },
                f,
            )

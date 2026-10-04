# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A tiny synthetic SHOW3D scene, so demos and tests run without downloading data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dataset import NUM_HAND_LANDMARKS, Show3DFrameRef


def build_synthetic_scene(root: Path, *, num_frames: int = 6) -> Path:
    """Fabricate a tiny SHOW3D scene (bundled ``mug`` object) plus a frame
    manifest under ``root``, and return the manifest path.

    The left hand has 21 landmarks near the object; the right hand is absent.
    Frame 3 is flagged as a headset tracking failure, to show that invalid frames
    are skipped. Each manifest row also carries the frame's ``sample_id``.
    """
    subject_id = "S000"
    scene_id = "mug_grab_demo"  # object alias -> "mug" -> bundled mug.glb
    rng = np.random.default_rng(0)

    scene_dir = root / "scenes" / subject_id / scene_id
    (scene_dir / "camera_calibration").mkdir(parents=True)
    (scene_dir / "metadata").mkdir()
    (scene_dir / "headset0.mp4").write_bytes(b"")
    (scene_dir / "headset1.mp4").write_bytes(b"")
    (scene_dir / "metadata" / "frame_info.json").write_text("{}")

    identity = [[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]
    by_index = {
        str(i): {
            "index": i,
            "T_WorldFromCamera": identity,
            # Frame 3 stands in for a headset-tracking failure.
            "is_synthesized": i == 3,
        }
        for i in range(num_frames)
    }
    for name in ("headset0", "headset1"):
        (scene_dir / "camera_calibration" / f"{name}.json").write_text(
            json.dumps(
                {
                    "ImageSizeX": 1408,
                    "ImageSizeY": 1408,
                    "fx": 600.0,
                    "fy": 600.0,
                    "cx": 704.0,
                    "cy": 704.0,
                    "DistortionModel": "PinholePlane",
                    "T_WorldFromCamera_by_index": by_index,
                }
            )
        )

    object_pose = {
        str(i): {
            "confidence": 0.99,
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [[0.0], [0.0], [0.0]],
        }
        for i in range(num_frames)
    }
    _write_json(
        root / "object_pose" / "v1" / "scenes" / subject_id / scene_id,
        "object_pose.json",
        object_pose,
    )

    # Left hand: 21 landmarks scattered near the object; right hand: absent.
    hand_pose = {
        str(i): {
            "hand_poses": {
                "0": {
                    "confidence": 0.99,
                    "landmarks_3d_mm": (
                        rng.standard_normal((NUM_HAND_LANDMARKS, 3)) * 50.0
                    ).tolist(),
                },
                "1": {"confidence": 0.0, "landmarks_3d_mm": None},
            }
        }
        for i in range(num_frames)
    }
    _write_json(
        root / "hand_pose" / "v2" / "scenes" / subject_id / scene_id,
        "hand_pose.json",
        hand_pose,
    )

    manifest_path = root / "manifest.jsonl"
    with manifest_path.open("w") as f:
        for i in range(num_frames):
            frame = Show3DFrameRef(
                subject_id=subject_id,
                scene_id=scene_id,
                frame_index=i,
                object_alias="mug",
            )
            row = {"sample_id": frame.sample_id, **frame.to_json()}
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")
    return manifest_path


def _write_json(directory: Path, name: str, payload: object) -> None:
    directory.mkdir(parents=True)
    (directory / name).write_text(json.dumps(payload))

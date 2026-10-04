# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generic dataloader helpers for the SHOW3D public starter repo.

This module knows the released dataset layout and annotation schemas, but it is
not tied to any benchmark task. Task packages should build on this API instead
of reimplementing path resolution or JSON parsing.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast, Literal

import cv2
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
CalibrationPoseSource = Literal[
    "mocap",
    "vio",
    "endpoint_interpolation",
    "smooth_mocap_interpolation",
    "legacy_interpolation",
    "legacy_unspecified",
]

DEFAULT_VIDEO_FPS: float = 60.0
# The high-confidence gate, for labels and training data.
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5
# The accept gate for hand poses.
ACCEPT_CONFIDENCE_THRESHOLD: float = 0.1
DEFAULT_HAND_POSE_VERSION: str = "v2"
# hand_pose versions whose per-scene UmeTrack file is `hand_pose.json`; from v3
# on it is `hand_pose_umetrack.json`, next to the other hand models' files.
LEGACY_HAND_POSE_VERSIONS: frozenset[str] = frozenset({"v1", "v2"})
DEFAULT_OBJECT_POSE_VERSION: str = "v1"
EGOCENTRIC_VIEWS: tuple[str, str] = ("headset0", "headset1")
POSE_CONTRACT_VERSION: int = 1
RIGID_TRANSFORM_TOLERANCE: float = 1e-4

LEFT_HAND_ID: str = "0"
RIGHT_HAND_ID: str = "1"
# Landmarks per hand in the UmeTrack / HOT3D skeleton of SHOW3D hand_pose.
NUM_HAND_LANDMARKS: int = 21
LEFT_HAND: str = "left"
RIGHT_HAND: str = "right"

# SHOW3D object alias -> HOT3D object LID, for the 22 objects shared with HOT3D
# (https://github.com/facebookresearch/hot3d). Load the CAD mesh from the HOT3D
# release by this LID, then pose it with ``ObjectPoseFrame.pose_vertices``. The 5
# SHOW3D-only aliases (keyboard2, cancoke, windex, clock, mug3) have no HOT3D
# mesh.
HOT3D_OBJECT_LID: dict[str, int] = {
    "dumbbell": 1,
    "mouse": 3,
    "keyboard": 4,
    "mug": 5,
    "mug2": 6,
    "balandabowl": 8,
    "vase": 11,
    "brushholder": 15,
    "birdhousetoy": 17,
    "dinotoy": 18,
    "whiteboardmarker": 21,
    "milk": 22,
    "orangejuice": 23,
    "mustard": 24,
    "ranch": 25,
    "bbq": 26,
    "cansoup": 27,
    "canparmesan": 28,
    "cantomatosauce": 29,
    "waffles": 30,
    "vegetables": 31,
    "aria": 33,
}


@dataclass(frozen=True)
class Show3DFrameRef:
    """One frame in one SHOW3D scene."""

    subject_id: str
    scene_id: str
    frame_index: int
    object_alias: str | None = None

    @classmethod
    def from_json(cls, row: Mapping[str, object]) -> "Show3DFrameRef":
        scene_id = _required_str(row, "scene_id")
        return cls(
            subject_id=_required_str(row, "subject_id"),
            scene_id=scene_id,
            frame_index=_required_int(row, "frame_index"),
            object_alias=_optional_str(row, "object_alias")
            or object_alias_from_scene_id(scene_id),
        )

    def to_json(self) -> dict[str, object]:
        row: dict[str, object] = {
            "subject_id": self.subject_id,
            "scene_id": self.scene_id,
            "frame_index": self.frame_index,
        }
        if self.object_alias is not None:
            row["object_alias"] = self.object_alias
        return row

    @property
    def sample_id(self) -> str:
        """``SUBJECT/SCENE:FRAME``, the frame key of manifests and frame indexes."""
        return f"{self.subject_id}/{self.scene_id}:{self.frame_index:06d}"


@dataclass(frozen=True)
class ObjectPoseFrame:
    """Object 6DoF pose (world-from-object) for a single frame.

    SHOW3D releases the object pose as ``rotation``/``translation_mm`` only; the
    object mesh comes from the HOT3D release
    (https://github.com/facebookresearch/hot3d), keyed by the object's HOT3D LID
    (see ``HOT3D_OBJECT_LID``). Pass the HOT3D mesh vertices (object frame, mm)
    to :meth:`pose_vertices` to get the object point cloud in world space.
    """

    confidence: float
    rotation: FloatArray  # (3, 3) world-from-object rotation
    translation_mm: FloatArray  # (3,) world-from-object translation, millimeters

    def pose_vertices(self, vertices_object_mm: FloatArray) -> FloatArray:
        """Transform object-frame vertices (Nx3, mm) into world space (Nx3, mm)."""
        vertices = np.asarray(vertices_object_mm, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Expected an Nx3 vertex array, got {vertices.shape}")
        return vertices @ self.rotation.T + self.translation_mm


@dataclass(frozen=True)
class HandPoseFrame:
    """One hand's pose for a single frame."""

    confidence: float
    landmarks_world_mm: FloatArray | None


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole calibration for one view at one frame.

    ``t_world_from_camera`` is ``None`` when this camera's transform is missing
    or the frame-level headset pose is invalid. ``is_synthesized`` retains its
    legacy meaning; ``is_pose_valid`` is the version 1 headset-pose verdict.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int
    t_world_from_camera: FloatArray | None
    is_synthesized: bool
    pose_source: CalibrationPoseSource = "legacy_unspecified"
    is_pose_valid: bool = False
    pose_contract_version: int | None = None


@dataclass(frozen=True)
class ViewFrame:
    """One egocentric view for a frame.

    ``video_path`` is always set. ``image`` is the decoded RGB frame
    ``(H, W, 3)`` uint8 when the dataset is built with ``decode_images=True``,
    else ``None``. ``calibration`` is the parsed pinhole calibration.
    """

    name: str
    video_path: Path
    image: NDArray[np.uint8] | None
    calibration: CameraCalibration | None


@dataclass(frozen=True)
class Show3DFrameData:
    """Generic dataloader item for one SHOW3D frame."""

    frame: Show3DFrameRef
    scene_dir: Path
    # Selected egocentric views keyed by name ("headset0" / "headset1"): one entry
    # for single-view, two for multi-view (chosen at dataset construction).
    views: dict[str, ViewFrame]
    frame_info_path: Path
    object_pose_path: Path
    hand_pose_path: Path
    object_pose: ObjectPoseFrame | None
    left_hand: HandPoseFrame | None
    right_hand: HandPoseFrame | None
    # Preserves the legacy aggregate over `is_synthesized` for compatibility.
    headset_tracking_valid: bool

    @property
    def headset_pose_valid(self) -> bool:
        """Frame-level headset-pose validity."""
        # Both camera files repeat this frame-level field; headset0 is canonical.
        headset_view = self.views.get(EGOCENTRIC_VIEWS[0])
        return (
            headset_view is not None
            and headset_view.calibration is not None
            and headset_view.calibration.is_pose_valid
        )


class Show3DPaths:
    """Path resolver for the public HuggingFace SHOW3D layout."""

    def __init__(
        self,
        root: str | Path,
        hand_pose_version: str = DEFAULT_HAND_POSE_VERSION,
        object_pose_version: str = DEFAULT_OBJECT_POSE_VERSION,
    ) -> None:
        self.root: Path = Path(root)
        self.hand_pose_version: str = hand_pose_version
        self.object_pose_version: str = object_pose_version

    def scene_dir(self, frame: Show3DFrameRef) -> Path:
        return self.root / "scenes" / frame.subject_id / frame.scene_id

    def headset_path(self, frame: Show3DFrameRef, camera_index: int) -> Path:
        if camera_index not in (0, 1):
            raise ValueError("SHOW3D public API exposes headset0/headset1 only")
        return self.scene_dir(frame) / f"headset{camera_index}.mp4"

    def frame_info_path(self, frame: Show3DFrameRef) -> Path:
        return self.scene_dir(frame) / "metadata" / "frame_info.json"

    def camera_calibration_path(self, frame: Show3DFrameRef, camera_name: str) -> Path:
        return self.scene_dir(frame) / "camera_calibration" / f"{camera_name}.json"

    def object_pose_path(self, frame: Show3DFrameRef) -> Path:
        return (
            self.root
            / "object_pose"
            / self.object_pose_version
            / "scenes"
            / frame.subject_id
            / frame.scene_id
            / "object_pose.json"
        )

    def hand_pose_path(self, frame: Show3DFrameRef) -> Path:
        return (
            self.root
            / "hand_pose"
            / self.hand_pose_version
            / "scenes"
            / frame.subject_id
            / frame.scene_id
            / (
                "hand_pose.json"
                if self.hand_pose_version in LEGACY_HAND_POSE_VERSIONS
                else "hand_pose_umetrack.json"
            )
        )


class Show3DDataset:
    """Generic frame-level dataset over a local SHOW3D mirror.

    It returns file paths plus optional object/hand pose annotations. It does
    not decode video frames, so downstream code can choose PyAV, torchvision,
    decord, or any other reader without changing the dataset contract.
    """

    def __init__(
        self,
        root: str | Path,
        frames: Iterable[Show3DFrameRef],
        *,
        hand_pose_version: str = DEFAULT_HAND_POSE_VERSION,
        object_pose_version: str = DEFAULT_OBJECT_POSE_VERSION,
        load_poses: bool = True,
        multiview: bool = True,
        decode_images: bool = False,
    ) -> None:
        self.paths: Show3DPaths = Show3DPaths(
            root,
            hand_pose_version=hand_pose_version,
            object_pose_version=object_pose_version,
        )
        self.frames: list[Show3DFrameRef] = list(frames)
        self.load_poses: bool = load_poses
        # Multi-view exposes both headsets; single-view exposes headset0 only.
        self.views: tuple[str, ...] = (
            EGOCENTRIC_VIEWS if multiview else EGOCENTRIC_VIEWS[:1]
        )
        # Decode the frame pixels (needs opencv) vs return only the video path.
        self.decode_images: bool = decode_images
        self._object_pose_cache: dict[Path, Mapping[str, object]] = {}
        self._hand_pose_cache: dict[Path, Mapping[str, object]] = {}
        self._calibration_cache: dict[Path, Mapping[str, object]] = {}

    @classmethod
    def from_manifest_jsonl(
        cls,
        root: str | Path,
        manifest_path: str | Path,
        *,
        hand_pose_version: str = DEFAULT_HAND_POSE_VERSION,
        object_pose_version: str = DEFAULT_OBJECT_POSE_VERSION,
        load_poses: bool = True,
        multiview: bool = True,
        decode_images: bool = False,
    ) -> "Show3DDataset":
        return cls(
            root,
            read_frame_manifest_jsonl(manifest_path),
            hand_pose_version=hand_pose_version,
            object_pose_version=object_pose_version,
            load_poses=load_poses,
            multiview=multiview,
            decode_images=decode_images,
        )

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> Show3DFrameData:
        return self.frame_data_for_ref(self.frames[index])

    def frame_data_for_ref(self, frame: Show3DFrameRef) -> Show3DFrameData:
        object_pose_path = self.paths.object_pose_path(frame)
        hand_pose_path = self.paths.hand_pose_path(frame)
        object_pose: ObjectPoseFrame | None = None
        left_hand: HandPoseFrame | None = None
        right_hand: HandPoseFrame | None = None
        if self.load_poses:
            object_pose = load_object_pose_frame(
                object_pose_path,
                frame.frame_index,
                cache=self._object_pose_cache,
            )
            hand_poses = load_hand_pose_frame(
                hand_pose_path,
                frame.frame_index,
                cache=self._hand_pose_cache,
            )
            left_hand = hand_poses.get(LEFT_HAND)
            right_hand = hand_poses.get(RIGHT_HAND)
        views: dict[str, ViewFrame] = {}
        for name in self.views:
            video_path = self.paths.headset_path(frame, int(name[-1]))
            views[name] = ViewFrame(
                name=name,
                video_path=video_path,
                image=(
                    _decode_frame(video_path, frame.frame_index)
                    if self.decode_images
                    else None
                ),
                calibration=load_camera_calibration(
                    self.paths.camera_calibration_path(frame, name),
                    frame.frame_index,
                    cache=self._calibration_cache,
                ),
            )
        headset_tracking_valid = not any(
            view.calibration is not None and view.calibration.is_synthesized
            for view in views.values()
        )
        return Show3DFrameData(
            frame=frame,
            scene_dir=self.paths.scene_dir(frame),
            views=views,
            frame_info_path=self.paths.frame_info_path(frame),
            object_pose_path=object_pose_path,
            hand_pose_path=hand_pose_path,
            object_pose=object_pose,
            left_hand=left_hand,
            right_hand=right_hand,
            headset_tracking_valid=headset_tracking_valid,
        )


def scene_activity(scene_id: str) -> str:
    parts = scene_id.split("_")
    if len(parts) < 3:
        return scene_id
    obj = parts[0]
    action = "_".join(parts[1:-1])
    return f"{obj}-{action}" if action else obj


def object_alias_from_scene_id(scene_id: str) -> str:
    return scene_id.split("_", 1)[0]


def hot3d_lid_for_scene(scene_id: str) -> int | None:
    """HOT3D object LID for a scene, or ``None`` for the 5 SHOW3D-only objects."""
    return HOT3D_OBJECT_LID.get(object_alias_from_scene_id(scene_id))


_OBJECT_MESHES_DIR: Path = Path(__file__).resolve().parent / "assets" / "objects"
_object_mesh_cache: dict[str, FloatArray | None] = {}


def _read_glb_vertices(path: Path) -> FloatArray:
    """Read POSITION vertices (Nx3, mm) from a binary glTF (.glb). stdlib-only."""
    data = path.read_bytes()
    if data[:4] != b"glTF":
        raise ValueError(f"Not a binary glTF (.glb): {path}")
    total = struct.unpack_from("<I", data, 8)[0]
    offset = 12
    gltf: dict[str, object] | None = None
    binary = b""
    while offset < total:
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == 0x4E4F534A:  # "JSON"
            gltf = json.loads(chunk)
        elif chunk_type == 0x004E4942:  # "BIN\0"
            binary = chunk
    if gltf is None:
        raise ValueError(f"GLB has no JSON chunk: {path}")
    accessors = cast(list, gltf["accessors"])
    views = cast(list, gltf["bufferViews"])
    parts: list[FloatArray] = []
    for mesh in cast(list, gltf.get("meshes", [])):
        for prim in cast(list, mesh["primitives"]):
            acc = accessors[prim["attributes"]["POSITION"]]
            view = views[acc["bufferView"]]
            start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
            count = acc["count"]
            parts.append(
                np.frombuffer(binary, np.float32, count * 3, start)
                .reshape(count, 3)
                .astype(np.float64)
            )
    return np.concatenate(parts, 0)


def load_object_mesh(object_alias: str) -> FloatArray | None:
    """Canonical object-mesh vertices (object frame, mm) for an alias, or None.

    Meshes are bundled under ``assets/objects/<alias>.glb``. Pose them with
    ``ObjectPoseFrame.pose_vertices`` and the released per-frame ``R``/``t`` to
    reconstruct the object point cloud in world space -- no HOT3D checkout and no
    multi-gigabyte per-frame vertices required.
    """
    if object_alias not in _object_mesh_cache:
        path = _OBJECT_MESHES_DIR / f"{object_alias}.glb"
        _object_mesh_cache[object_alias] = (
            _read_glb_vertices(path) if path.exists() else None
        )
    return _object_mesh_cache[object_alias]


def default_object_mesh_provider() -> Callable[[str], FloatArray | None]:
    """Object-alias -> canonical mesh provider backed by the bundled GLBs."""
    return load_object_mesh


def _rigid_transform_from_json(value: object) -> FloatArray | None:
    if value is None:
        return None
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Calibration transform must be numeric") from error
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("Calibration transform must be a finite 4x4 matrix")
    if not np.allclose(
        matrix[3],
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        rtol=0.0,
        atol=RIGID_TRANSFORM_TOLERANCE,
    ):
        raise ValueError("Calibration transform must be rigid")
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        rtol=0.0,
        atol=RIGID_TRANSFORM_TOLERANCE,
    ) or not np.isclose(
        np.linalg.det(rotation),
        1.0,
        rtol=0.0,
        atol=RIGID_TRANSFORM_TOLERANCE,
    ):
        raise ValueError("Calibration transform must be rigid")
    return matrix


def _pose_contract_version(data: Mapping[str, object]) -> int | None:
    if "pose_contract_version" not in data:
        return None
    value = data["pose_contract_version"]
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value != POSE_CONTRACT_VERSION
    ):
        raise ValueError(f"Unsupported calibration pose_contract_version: {value}")
    return value


def _pose_contract_fields(
    entry: Mapping[str, object],
    is_synthesized: bool,
    contract_version: int | None,
) -> tuple[CalibrationPoseSource, bool]:
    raw_pose_source = entry.get("pose_source")
    raw_pose_valid = entry.get("is_pose_valid")
    has_contract_fields = "pose_source" in entry or "is_pose_valid" in entry
    if contract_version is None:
        if has_contract_fields:
            raise ValueError(
                "Calibration pose_source and is_pose_valid require "
                "pose_contract_version 1"
            )
        return (
            "legacy_interpolation" if is_synthesized else "legacy_unspecified",
            not is_synthesized,
        )
    if raw_pose_source is None or not isinstance(raw_pose_valid, bool):
        raise ValueError(
            "Calibration pose contract v1 requires pose_source and is_pose_valid"
        )
    if raw_pose_source in {
        "mocap",
        "vio",
        "endpoint_interpolation",
        "smooth_mocap_interpolation",
        "legacy_unspecified",
    }:
        return cast(CalibrationPoseSource, raw_pose_source), raw_pose_valid
    else:
        raise ValueError(f"Unknown calibration pose_source: {raw_pose_source}")


def load_camera_calibration(
    path: str | Path,
    frame_index: int,
    *,
    cache: dict[Path, Mapping[str, object]] | None = None,
) -> CameraCalibration | None:
    """Pinhole intrinsics + this frame's world-from-camera extrinsics, or None."""
    cal_path = Path(path)
    if not cal_path.exists():
        return None
    data = _load_json_mapping(cal_path, cache)
    pose_contract_version = _pose_contract_version(data)
    t_world_from_camera: FloatArray | None = None
    is_synthesized = False
    pose_source: CalibrationPoseSource = "legacy_unspecified"
    is_pose_valid = False
    by_index = data.get("T_WorldFromCamera_by_index")
    if isinstance(by_index, dict):
        raw_entry = by_index.get(str(frame_index))
        if isinstance(raw_entry, dict):
            entry = cast(dict[str, object], raw_entry)
            raw_is_synthesized = entry.get("is_synthesized", False)
            if pose_contract_version is None:
                is_synthesized = bool(raw_is_synthesized)
            else:
                if "is_synthesized" not in entry or not isinstance(
                    raw_is_synthesized, bool
                ):
                    raise ValueError("Calibration is_synthesized must be a boolean")
                is_synthesized = raw_is_synthesized
            pose_source, is_pose_valid = _pose_contract_fields(
                entry,
                is_synthesized,
                pose_contract_version,
            )
            matrix = entry.get("T_WorldFromCamera")
            if pose_contract_version is None:
                matrix_array = (
                    np.asarray(matrix, dtype=np.float64)
                    if matrix is not None and not is_synthesized
                    else None
                )
            else:
                matrix_array = _rigid_transform_from_json(matrix)
            if is_pose_valid and matrix_array is not None:
                t_world_from_camera = matrix_array
    return CameraCalibration(
        fx=_optional_float(data, "fx", 0.0),
        fy=_optional_float(data, "fy", 0.0),
        cx=_optional_float(data, "cx", 0.0),
        cy=_optional_float(data, "cy", 0.0),
        image_width=_required_int(data, "ImageSizeX"),
        image_height=_required_int(data, "ImageSizeY"),
        t_world_from_camera=t_world_from_camera,
        is_synthesized=is_synthesized,
        pose_source=pose_source,
        is_pose_valid=is_pose_valid,
        pose_contract_version=pose_contract_version,
    )


def _decode_frame(video_path: Path, frame_index: int) -> NDArray[np.uint8]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = capture.read()
        if not ok:
            raise ValueError(f"Could not read frame {frame_index} from {video_path}")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def read_frame_manifest_jsonl(path: str | Path) -> list[Show3DFrameRef]:
    rows: list[Show3DFrameRef] = []
    with Path(path).open() as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            if not isinstance(raw, dict):
                raise ValueError(f"Manifest line {line_number} is not a JSON object")
            rows.append(Show3DFrameRef.from_json(cast(dict[str, object], raw)))
    return rows


def write_frame_manifest_jsonl(
    path: str | Path, frames: Iterable[Show3DFrameRef]
) -> None:
    with Path(path).open("w") as f:
        for frame in frames:
            f.write(json.dumps(frame.to_json(), sort_keys=True))
            f.write("\n")


def sampled_frame_indices(
    num_frames: int,
    sampling_fps: float,
    *,
    video_fps: float = DEFAULT_VIDEO_FPS,
    start_frame: int = 0,
) -> list[int]:
    """Deterministically subsample source-frame indices to ``sampling_fps``."""
    if num_frames <= 0:
        return []
    if sampling_fps <= 0.0:
        raise ValueError("sampling_fps must be positive")
    if video_fps <= 0.0:
        raise ValueError("video_fps must be positive")
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if start_frame >= num_frames:
        return []

    interval = video_fps / sampling_fps
    rounded_stride = round(interval)
    if rounded_stride > 0 and math.isclose(
        interval, float(rounded_stride), rel_tol=0.0, abs_tol=1e-6
    ):
        return list(range(start_frame, num_frames, int(rounded_stride)))

    out: list[int] = []
    seen: set[int] = set()
    k = 0
    while True:
        frame = int(round(start_frame + k * interval))
        if frame >= num_frames:
            break
        if frame not in seen:
            out.append(frame)
            seen.add(frame)
        k += 1
    return out


def load_object_pose_frame(
    path: str | Path,
    frame_index: int,
    *,
    cache: dict[Path, Mapping[str, object]] | None = None,
) -> ObjectPoseFrame | None:
    """This frame's object pose, or None when the frame or the file is missing."""
    pose_path = Path(path)
    if not pose_path.exists():
        return None
    data = _load_json_mapping(pose_path, cache)
    frame = data.get(str(frame_index))
    if not isinstance(frame, dict):
        return None
    frame_dict = cast(dict[str, object], frame)
    confidence = _optional_float(frame_dict, "confidence", 0.0)
    rotation_value = frame_dict.get("R")
    translation_value = frame_dict.get("t")
    if rotation_value is None or translation_value is None:
        return None
    rotation = np.asarray(rotation_value, dtype=np.float64)
    translation = np.asarray(translation_value, dtype=np.float64).reshape(-1)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        return None
    return ObjectPoseFrame(
        confidence=confidence,
        rotation=rotation,
        translation_mm=translation,
    )


def load_hand_pose_frame(
    path: str | Path,
    frame_index: int,
    *,
    cache: dict[Path, Mapping[str, object]] | None = None,
) -> dict[str, HandPoseFrame | None]:
    """This frame's hand poses by side; a hand is None when missing, as both are
    when the frame or the file is missing."""
    pose_path = Path(path)
    if not pose_path.exists():
        return {LEFT_HAND: None, RIGHT_HAND: None}
    data = _load_json_mapping(pose_path, cache)
    frame = data.get(str(frame_index))
    if not isinstance(frame, dict):
        return {LEFT_HAND: None, RIGHT_HAND: None}
    frame_dict = cast(dict[str, object], frame)
    raw_hand_poses = frame_dict.get("hand_poses")
    if not isinstance(raw_hand_poses, dict):
        return {LEFT_HAND: None, RIGHT_HAND: None}
    hand_poses = cast(dict[str, object], raw_hand_poses)
    return {
        LEFT_HAND: _parse_one_hand_pose(hand_poses.get(LEFT_HAND_ID)),
        RIGHT_HAND: _parse_one_hand_pose(hand_poses.get(RIGHT_HAND_ID)),
    }


def array_from_json(value: object, *, columns: int) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, columns), dtype=np.float64)
    if len(array.shape) != 2 or array.shape[1] != columns:
        raise ValueError(f"Expected an Nx{columns} array, got shape {array.shape}")
    return array


def _parse_one_hand_pose(raw_pose: object) -> HandPoseFrame | None:
    if not isinstance(raw_pose, dict):
        return None
    pose = cast(dict[str, object], raw_pose)
    confidence = _optional_float(pose, "confidence", 0.0)
    landmarks_value = pose.get("landmarks_3d_mm")
    landmarks = (
        array_from_json(landmarks_value, columns=3)
        if landmarks_value is not None
        else None
    )
    return HandPoseFrame(confidence=confidence, landmarks_world_mm=landmarks)


def _load_json_mapping(
    path: Path,
    cache: dict[Path, Mapping[str, object]] | None,
) -> Mapping[str, object]:
    if cache is not None and path in cache:
        return cache[path]
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    mapping = cast(dict[str, object], data)
    if cache is not None:
        cache[path] = mapping
    return mapping


def _required_str(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Required string field missing: {key}")
    return value


def _optional_str(row: Mapping[str, object], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Expected string field for {key}")
    return value


def _required_int(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Required int field missing: {key}")
    return value


def _optional_float(row: Mapping[str, object], key: str, default: float) -> float:
    value = row.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"Expected numeric field for {key}")
    return float(value)

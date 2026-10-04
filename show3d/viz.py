# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Visualization library for SHOW3D: hand skeletons, objects, and hand meshes.

Two renders of one frame, each writing a PNG:

* :func:`render_overlay`  -- hand skeleton + object projected onto the frame (2D),
  or onto both egocentric frames side by side.
* :func:`render_geometry` -- hand skeleton + object surface in 3D.

:func:`run_visualization` picks a suitable frame from a frame manifest and
dispatches to one of them. :func:`render_hand_meshes` draws the posed hand meshes
of one hand model (:mod:`show3d.hand_mesh`) on a frame or a frame range.
Projection lives in :mod:`show3d.camera`; the CLI is :mod:`show3d.demo_viz`.
Needs matplotlib.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from . import camera
from .dataset import (
    ACCEPT_CONFIDENCE_THRESHOLD,
    CameraCalibration,
    DEFAULT_CONFIDENCE_THRESHOLD,
    default_object_mesh_provider,
    DEFAULT_VIDEO_FPS,
    EGOCENTRIC_VIEWS,
    FloatArray,
    HandPoseFrame,
    LEFT_HAND,
    load_camera_calibration,
    object_alias_from_scene_id,
    RIGHT_HAND,
    Show3DDataset,
    Show3DFrameData,
    Show3DFrameRef,
    Show3DPaths,
)
from .hand_mesh import (
    DEFAULT_MESH_HAND_POSE_VERSION,
    HandMesh,
    HandMeshScene,
    LEFT_SLOT,
    RIGHT_SLOT,
)

MODES: tuple[str, ...] = ("overlay", "geometry")

# 21-joint UmeTrack / HOT3D hand landmark order used by SHOW3D hand_pose:
# fingertips are 0-4, the wrist is 5, and the palm center is 20 (not a bone
# endpoint). Each finger chain is wrist -> proximal -> intermediate -> distal ->
# fingertip; the last four edges are the palm arch across the knuckles.
HAND_EDGES: tuple[tuple[int, int], ...] = (
    (5, 17),
    (17, 18),
    (18, 19),
    (19, 4),  # pinky
    (5, 14),
    (14, 15),
    (15, 16),
    (16, 3),  # ring
    (5, 11),
    (11, 12),
    (12, 13),
    (13, 2),  # middle
    (5, 8),
    (8, 9),
    (9, 10),
    (10, 1),  # index
    (5, 6),
    (6, 7),
    (7, 0),  # thumb
    (6, 8),
    (8, 11),
    (11, 14),
    (14, 17),  # palm arch
)

# Hand RGB per slot, orange for the left hand and cyan for the right: the colors
# of the released viz_*.mp4 videos. Skeletons and meshes both use them.
MESH_COLORS: tuple[tuple[int, int, int], tuple[int, int, int]] = (
    (235, 104, 52),
    (42, 184, 214),
)


def _hex_color(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# (legend label, color) per hand, keyed by ``LEFT_HAND`` / ``RIGHT_HAND``.
HAND_STYLES: dict[str, tuple[str, str]] = {
    LEFT_HAND: ("left hand", _hex_color(MESH_COLORS[LEFT_SLOT])),
    RIGHT_HAND: ("right hand", _hex_color(MESH_COLORS[RIGHT_SLOT])),
}
# The ``view`` that overlays both egocentric views side by side.
STEREO_VIEW: str = "stereo"
# A vertex barely in front of the camera plane projects millions of pixels away.
# Pulling it to this distance from the principal point along its own direction
# keeps OpenCV's fixed-point coordinates in int32 and moves the triangle's edges
# by only a few pixels inside the image.
_MAX_PIXEL: float = 1e5
_SUBPIXEL_BITS: int = 4


# ----------------------------------------------------------------------------
# drawing primitives
# ----------------------------------------------------------------------------
def draw_object_3d(ax: Any, surface_mm: FloatArray, *, max_points: int = 4000) -> None:
    """Scatter the object surface points (thinned for a light plot)."""
    step = max(1, surface_mm.shape[0] // max_points)
    ax.scatter(
        surface_mm[::step, 0],
        surface_mm[::step, 1],
        surface_mm[::step, 2],
        s=2,
        c="0.6",
        label="object surface",
    )


def draw_hand_skeleton_3d(
    ax: Any, joints_mm: FloatArray, color: str, label: str
) -> None:
    """Draw the hand as connected bones plus joint markers, in 3D."""
    for a, b in HAND_EDGES:
        ax.plot(
            [joints_mm[a, 0], joints_mm[b, 0]],
            [joints_mm[a, 1], joints_mm[b, 1]],
            [joints_mm[a, 2], joints_mm[b, 2]],
            c=color,
            linewidth=2.0,
        )
    ax.scatter(
        joints_mm[:, 0],
        joints_mm[:, 1],
        joints_mm[:, 2],
        s=18,
        c=color,
        label=label,
        depthshade=False,
    )


def set_equal_aspect_3d(ax: Any, points_mm: FloatArray) -> None:
    """Make the 3D box proportional to the data so geometry is not distorted."""
    span = points_mm.max(axis=0) - points_mm.min(axis=0)
    ax.set_box_aspect(tuple(float(s) if s > 0 else 1.0 for s in span))


def draw_object_2d(ax: Any, object_uv: FloatArray, valid: NDArray[np.bool_]) -> None:
    """Overlay projected object-surface points on an image axis."""
    pts = object_uv[valid]
    ax.scatter(pts[:, 0], pts[:, 1], s=2, c="gold", alpha=0.35, label="object")


def draw_hand_skeleton_2d(
    ax: Any, joints_uv: FloatArray, valid: NDArray[np.bool_], color: str, label: str
) -> None:
    """Overlay the hand skeleton (bones + joints) on an image axis."""
    for a, b in HAND_EDGES:
        if valid[a] and valid[b]:
            ax.plot(
                [joints_uv[a, 0], joints_uv[b, 0]],
                [joints_uv[a, 1], joints_uv[b, 1]],
                c=color,
                linewidth=2.0,
            )
    shown = joints_uv[valid]
    ax.scatter(shown[:, 0], shown[:, 1], s=18, c=color, label=label)


def draw_hand_meshes(
    image_rgb: NDArray[np.uint8],
    meshes: Sequence[tuple[HandMesh, tuple[int, int, int]]],
    calibration: CameraCalibration,
    *,
    alpha: float = 0.7,
) -> NDArray[np.uint8]:
    """Draw ``(mesh, rgb)`` pairs on a copy of the frame as shaded triangles.

    Triangles of all meshes are painted together from far to near, so a hand
    occludes the other. A triangle with a vertex behind the camera is skipped.
    """
    t_world_from_camera = calibration.t_world_from_camera
    if t_world_from_camera is None:
        raise ValueError("frame has no valid t_world_from_camera")
    depths: list[FloatArray] = []
    points: list[NDArray[np.int32]] = []
    colors: list[FloatArray] = []
    for mesh, rgb in meshes:
        triangles = camera.world_to_camera(mesh.vertices_world_mm, t_world_from_camera)[
            mesh.faces
        ]
        front = (triangles[:, :, 2] > 0.0).all(axis=1)
        triangles = triangles[front]
        uv, _valid = camera.project_to_image(mesh.vertices_world_mm, calibration)
        uv = uv[mesh.faces[front]]
        center = np.array([calibration.cx, calibration.cy])
        offset = uv - center
        distance = np.linalg.norm(offset, axis=-1, keepdims=True)
        uv = np.where(
            distance > _MAX_PIXEL,
            center + offset * (_MAX_PIXEL / np.maximum(distance, _MAX_PIXEL)),
            uv,
        )
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        centers = triangles.mean(axis=1)
        # Lambert shading with the light at the camera.
        cosine = np.abs((normals * centers).sum(axis=1)) / np.maximum(
            np.linalg.norm(normals, axis=1) * np.linalg.norm(centers, axis=1), 1e-12
        )
        depths.append(centers[:, 2])
        points.append(np.round(uv * (1 << _SUBPIXEL_BITS)).astype(np.int32))
        colors.append((0.3 + 0.7 * cosine)[:, None] * np.asarray(rgb, np.float64))
    out = image_rgb.copy()
    if not depths:
        return out
    depth = np.concatenate(depths)
    triangle_points = np.concatenate(points)
    triangle_colors = np.concatenate(colors)
    painted = image_rgb.copy()
    covered = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    for index in np.argsort(-depth):
        cv2.fillConvexPoly(
            painted,
            triangle_points[index],
            triangle_colors[index].tolist(),
            shift=_SUBPIXEL_BITS,
        )
        cv2.fillConvexPoly(covered, triangle_points[index], 255, shift=_SUBPIXEL_BITS)
    mask = covered > 0
    out[mask] = (alpha * painted[mask] + (1.0 - alpha) * image_rgb[mask]).astype(
        np.uint8
    )
    return out


# ----------------------------------------------------------------------------
# frame -> geometry
# ----------------------------------------------------------------------------
def object_surface(frame_data: Show3DFrameData) -> FloatArray | None:
    """The object's canonical mesh posed into world space, or None."""
    object_pose = frame_data.object_pose
    frame = frame_data.frame
    alias = frame.object_alias or object_alias_from_scene_id(frame.scene_id)
    mesh = default_object_mesh_provider()(alias)
    if mesh is None or object_pose is None:
        return None
    return object_pose.pose_vertices(mesh)


def frame_hands(frame_data: Show3DFrameData) -> dict[str, HandPoseFrame | None]:
    """The frame's hand poses keyed by ``LEFT_HAND`` / ``RIGHT_HAND``."""
    return {LEFT_HAND: frame_data.left_hand, RIGHT_HAND: frame_data.right_hand}


def drawn_landmarks(hand: HandPoseFrame | None) -> FloatArray | None:
    """The hand's landmarks to draw, or None when the hand is missing, has no
    landmarks, or is at or below the accept gate that :mod:`show3d.hand_mesh`
    also uses."""
    if hand is None or hand.confidence <= ACCEPT_CONFIDENCE_THRESHOLD:
        return None
    return hand.landmarks_world_mm


def present_hands(frame_data: Show3DFrameData) -> list[tuple[str, str, FloatArray]]:
    """``(label, color, joints_world_mm)`` for each hand with
    :func:`drawn_landmarks`."""
    out: list[tuple[str, str, FloatArray]] = []
    for side, hand in frame_hands(frame_data).items():
        joints = drawn_landmarks(hand)
        if joints is None:
            continue
        label, color = HAND_STYLES[side]
        out.append((label, color, joints))
    return out


def finish_3d(ax: Any, extent: list[FloatArray], title: str) -> None:
    """Fit the 3D box to ``extent``, set the view, title, axis labels, legend."""
    if extent:
        set_equal_aspect_3d(ax, np.concatenate(extent, axis=0))
    ax.view_init(elev=18, azim=-70)
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.legend(loc="upper right", fontsize="small")


# ----------------------------------------------------------------------------
# renders
# ----------------------------------------------------------------------------
def render_geometry(frame_data: Show3DFrameData, out_path: str | Path) -> None:
    """3D hand skeleton + object surface."""
    fig = Figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    extent: list[FloatArray] = []
    surface = object_surface(frame_data)
    if surface is not None:
        draw_object_3d(ax, surface)
        extent.append(surface)
    for label, color, joints in present_hands(frame_data):
        draw_hand_skeleton_3d(ax, joints, color, label)
        extent.append(joints)
    finish_3d(ax, extent, f"SHOW3D hand + object: {frame_data.frame.sample_id}")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")


def _decode_frame(video_path: Path, frame_index: int) -> FloatArray | None:
    if not video_path.exists():
        return None
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def _overlay_view_names(view_name: str) -> tuple[str, ...]:
    """The egocentric views an overlay of ``view_name`` draws: both for
    ``STEREO_VIEW``, else just ``view_name``."""
    return EGOCENTRIC_VIEWS if view_name == STEREO_VIEW else (view_name,)


def _overlay_view(
    frame_data: Show3DFrameData, view_name: str
) -> tuple[Path, CameraCalibration]:
    view = frame_data.views.get(view_name)
    if view is None or view.calibration is None:
        raise ValueError(f"view {view_name!r} has no calibration to overlay")
    if view.calibration.t_world_from_camera is None:
        raise ValueError(f"view {view_name!r} has no valid t_world_from_camera")
    return view.video_path, view.calibration


def _draw_overlay(
    ax: Any,
    frame_data: Show3DFrameData,
    video_path: Path,
    calibration: CameraCalibration,
) -> None:
    image = _decode_frame(video_path, frame_data.frame.frame_index)
    if image is not None:
        ax.imshow(image)
    else:
        ax.set_facecolor("0.9")

    surface = object_surface(frame_data)
    if surface is not None:
        object_uv, object_valid = camera.project_to_image(surface, calibration)
        draw_object_2d(ax, object_uv, object_valid)
    for label, color, joints in present_hands(frame_data):
        joints_uv, valid = camera.project_to_image(joints, calibration)
        if bool(valid.any()):
            draw_hand_skeleton_2d(ax, joints_uv, valid, color, label)

    ax.set_xlim(0, calibration.image_width)
    ax.set_ylim(calibration.image_height, 0)  # image row 0 at top
    ax.set_aspect("equal")
    ax.axis("off")


def render_overlay(
    frame_data: Show3DFrameData, view_name: str, out_path: str | Path
) -> None:
    """Hand skeleton + object projected onto the egocentric frame (2D). With
    ``STEREO_VIEW``, onto both egocentric frames side by side."""
    names = _overlay_view_names(view_name)
    views = [_overlay_view(frame_data, name) for name in names]
    first = views[0][1]
    fig = Figure(
        figsize=(8 * len(views), 8 * first.image_height / first.image_width),
        layout="compressed",
    )
    axes: Any = fig.subplots(1, len(views), squeeze=False)[0]
    sample_id = frame_data.frame.sample_id
    for ax, name, (video_path, calibration) in zip(axes, names, views):
        _draw_overlay(ax, frame_data, video_path, calibration)
        ax.set_title(name if len(views) > 1 else f"SHOW3D {name} overlay: {sample_id}")
    if len(views) > 1:
        fig.suptitle(f"SHOW3D stereo overlay: {sample_id}")
    axes[-1].legend(loc="upper right", fontsize="small")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")


# ----------------------------------------------------------------------------
# frame selection + entry point
# ----------------------------------------------------------------------------
def _has_confident_hand(frame_data: Show3DFrameData) -> bool:
    return any(
        hand is not None
        and hand.landmarks_world_mm is not None
        and hand.confidence > DEFAULT_CONFIDENCE_THRESHOLD
        for hand in frame_hands(frame_data).values()
    )


def _has_view_pose(frame_data: Show3DFrameData, view_name: str) -> bool:
    view = frame_data.views.get(view_name)
    return (
        view is not None
        and view.calibration is not None
        and view.calibration.t_world_from_camera is not None
    )


def _projects_into_view(frame_data: Show3DFrameData, view_name: str) -> bool:
    view = frame_data.views.get(view_name)
    if view is None or view.calibration is None:
        return False
    if view.calibration.t_world_from_camera is None:
        return False
    for _label, _color, joints in present_hands(frame_data):
        _uv, valid = camera.project_to_image(joints, view.calibration)
        if int(valid.sum()) >= 8:
            return True
    return False


def _pick_frame(
    dataset: Show3DDataset, mode: str, view_name: str
) -> Show3DFrameData | None:
    """The first frame with a valid headset pose and a confident hand; for
    overlay, preferably one whose hand lands in every overlaid view."""
    names = _overlay_view_names(view_name)
    fallback: Show3DFrameData | None = None
    for index in range(len(dataset)):
        frame_data = dataset[index]
        if not frame_data.headset_pose_valid or not _has_confident_hand(frame_data):
            continue
        if mode != "overlay":
            return frame_data
        if not all(_has_view_pose(frame_data, name) for name in names):
            continue
        if fallback is None:
            fallback = frame_data
        if all(_projects_into_view(frame_data, name) for name in names):
            return frame_data
    return fallback


def run_visualization(
    root: str | Path,
    manifest_path: str | Path,
    out_path: str | Path,
    *,
    mode: str = "geometry",
    view: str = "headset0",
    verbose: bool = False,
) -> Path:
    """Pick a suitable frame from the manifest and render ``mode`` to ``out_path``."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    dataset = Show3DDataset.from_manifest_jsonl(root, manifest_path)
    frame_data = _pick_frame(dataset, mode, view)
    if frame_data is None:
        if not dataset.frames:
            raise ValueError(f"{manifest_path} lists no frames")
        raise ValueError(
            f"no frame in {manifest_path} has a valid headset pose and a hand above "
            f"confidence {DEFAULT_CONFIDENCE_THRESHOLD} (hand poses read from "
            f"{dataset.paths.hand_pose_path(dataset.frames[0])}, for example)"
        )

    out = Path(out_path)
    if mode == "overlay":
        render_overlay(frame_data, view, out)
    else:
        render_geometry(frame_data, out)
    if verbose:
        print(f"[{mode}] {frame_data.frame.sample_id} -> {out}")
    return out


def _hand_meshes(
    scene: HandMeshScene, frame_index: int
) -> list[tuple[HandMesh, tuple[int, int, int]]]:
    meshes: list[tuple[HandMesh, tuple[int, int, int]]] = []
    for slot in (LEFT_SLOT, RIGHT_SLOT):
        mesh = scene.mesh(frame_index, slot)
        if mesh is not None:
            meshes.append((mesh, MESH_COLORS[slot]))
    return meshes


def _start_frame(scene: HandMeshScene, frame_index: int | None) -> int:
    """``frame_index``, or the scene's first frame with a hand mesh."""
    if frame_index is None:
        for key in sorted(scene.frames, key=int):
            if _hand_meshes(scene, int(key)):
                return int(key)
        raise ValueError("the scene has no hand mesh")
    if frame_index < 0:
        raise ValueError(f"frame_index must be non-negative, got {frame_index}")
    return frame_index


def _overlay_hand_meshes(
    scene: HandMeshScene,
    frame_index: int,
    frame_bgr: NDArray[np.uint8],
    calibration: CameraCalibration | None,
) -> NDArray[np.uint8]:
    hands = _hand_meshes(scene, frame_index)
    if not hands or calibration is None or calibration.t_world_from_camera is None:
        return frame_bgr
    image = draw_hand_meshes(
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), hands, calibration
    )
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _open_video_writer(
    out: Path, capture: cv2.VideoCapture, frame_bgr: NDArray[np.uint8]
) -> cv2.VideoWriter:
    height, width = frame_bgr.shape[:2]
    fps = capture.get(cv2.CAP_PROP_FPS) or DEFAULT_VIDEO_FPS
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"could not open {out} for writing")
    return writer


def _write_frames(
    scene: HandMeshScene,
    video_path: Path,
    calibration_path: Path,
    out: Path,
    frame_index: int,
    num_frames: int | None,
) -> int:
    """Draw and write the frames; return how many were written."""
    calibration_cache: dict[Path, Mapping[str, object]] = {}
    capture = cv2.VideoCapture(str(video_path))
    writer: cv2.VideoWriter | None = None
    written = 0
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        for index in range(frame_index, frame_index + (num_frames or 1)):
            ok, frame_bgr = capture.read()
            if not ok:
                break
            calibration = load_camera_calibration(
                calibration_path, index, cache=calibration_cache
            )
            frame_bgr = _overlay_hand_meshes(scene, index, frame_bgr, calibration)
            if num_frames is None:
                if not cv2.imwrite(str(out), frame_bgr):
                    raise OSError(f"could not write {out}")
            else:
                if writer is None:
                    writer = _open_video_writer(out, capture, frame_bgr)
                writer.write(frame_bgr)
            written += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return written


def render_hand_meshes(
    root: str | Path,
    subject: str,
    scene: str,
    model: str,
    out_path: str | Path,
    *,
    view: str = "headset0",
    frame_index: int | None = None,
    num_frames: int | None = None,
    asset_dir: str | Path | None = None,
    hand_pose_version: str = DEFAULT_MESH_HAND_POSE_VERSION,
    verbose: bool = False,
) -> Path:
    """Draw ``model``'s hand meshes on frames of ``view``.

    With ``num_frames``, write that many frames from ``frame_index`` to an MP4;
    without it, write frame ``frame_index`` to an image. With no
    ``frame_index``, start at the scene's first frame with a hand mesh.
    """
    if num_frames is not None and num_frames < 1:
        raise ValueError(f"num_frames must be at least 1, got {num_frames}")
    meshes = HandMeshScene(
        root, subject, scene, model, asset_dir=asset_dir, version=hand_pose_version
    )
    start = _start_frame(meshes, frame_index)
    paths = Show3DPaths(root)
    frame = Show3DFrameRef(subject_id=subject, scene_id=scene, frame_index=start)
    video_path = paths.headset_path(frame, EGOCENTRIC_VIEWS.index(view))
    out = Path(out_path)
    written = _write_frames(
        meshes,
        video_path,
        paths.camera_calibration_path(frame, view),
        out,
        start,
        num_frames,
    )
    if written == 0:
        raise ValueError(f"could not read frame {start} of {video_path}")
    if verbose:
        print(f"[{model}] {subject}/{scene} {view} frames {start}+{written} -> {out}")
    return out

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Extract SHOW3D egocentric frames to a random-access image store, at a chosen fps.

Random per-frame seeking into the MP4s is slow, so training from video directly
is decode-bound. This tool decodes each recording's headset video **once,
sequentially** (no seeking), keeps the frames at your target fps, and writes them
as images plus an ``index.jsonl`` that a fast, shuffle-friendly map-style dataset
can read.

Frames are written as JPEG by default. The source videos are already lossy
(H.264/H.265), so JPEG at high quality adds negligible loss at a fraction of PNG's
size; use ``--format png`` only if you need bit-exact decoded frames.

By default every scene under ``--root`` is extracted. ``--manifest`` restricts the
run to the scenes a JSONL manifest names, and ``--require-object-pose`` to the
scenes with object-pose ground truth.

Sampling keeps every k-th frame, so ``--fps`` must be a whole number that divides
the 60 fps source: 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, or 60. It is required
and rejects any other value.

    python -m show3d.extract_images --root /path/to/show3d --out frames/ --fps 10
    python -m show3d.extract_images --root /path/to/show3d --out frames/ --fps 5 \\
        --views headset0 --quality 80 --workers 16 --posed-only

Output layout::

    <out>/<subject>/<scene>/<view>/<frame_index:06d>.jpg
    <out>/index.jsonl          # one row per (frame, view): sample_id, subject, scene, frame, view, image
    <out>/extract_info.json    # the parameters used

``sample_id`` is ``SUBJECT/SCENE:FRAME`` and is the same for both views of a frame,
so per-frame annotations can join to the images on it.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from concurrent.futures import as_completed, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2

from .dataset import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_OBJECT_POSE_VERSION,
    EGOCENTRIC_VIEWS,
    load_object_pose_frame,
    sampled_frame_indices,
    Show3DFrameRef,
    Show3DPaths,
)

DEFAULT_SOURCE_FPS: float = 60.0  # SHOW3D is captured at 60 fps


@dataclass(frozen=True)
class ExtractConfig:
    """Per-run extraction settings (passed to each worker; must stay picklable)."""

    fps: int
    views: tuple[str, ...]
    image_format: str
    quality: int
    posed_only: bool
    object_pose_version: str


def valid_fps_values(source_fps: float = DEFAULT_SOURCE_FPS) -> list[int]:
    """Whole-number fps that divide the source rate (so the stride is an integer)."""
    source = int(round(source_fps))
    return [k for k in range(1, source + 1) if source % k == 0]


def validate_fps(fps: int, source_fps: float = DEFAULT_SOURCE_FPS) -> None:
    """Reject an fps that is not a whole divisor of the source rate."""
    valid = valid_fps_values(source_fps)
    if fps not in valid:
        raise ValueError(
            f"--fps {fps} is not achievable from a {int(round(source_fps))} fps "
            f"source: it must be a whole number that divides evenly. "
            f"Valid values: {valid}."
        )


def _encode_params(image_format: str, quality: int) -> tuple[str, list[int]]:
    """Return the file extension and cv2.imwrite params for a format."""
    if image_format == "png":
        return ".png", [cv2.IMWRITE_PNG_COMPRESSION, 6]
    return ".jpg", [cv2.IMWRITE_JPEG_QUALITY, quality]


def iter_scenes(root: Path, subjects: set[str] | None) -> list[tuple[str, str]]:
    """List ``(subject, scene)`` pairs under ``<root>/scenes/``."""
    scenes_root = root / "scenes"
    pairs: list[tuple[str, str]] = []
    if not scenes_root.is_dir():
        raise FileNotFoundError(f"No scenes/ directory under root: {scenes_root}")
    for subject_dir in sorted(scenes_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        if subjects is not None and subject_dir.name not in subjects:
            continue
        for scene_dir in sorted(subject_dir.iterdir()):
            if scene_dir.is_dir():
                pairs.append((subject_dir.name, scene_dir.name))
    return pairs


def scenes_from_manifest(manifest_path: str | Path) -> list[tuple[str, str]]:
    """Distinct ``(subject_id, scene_id)`` from a JSONL manifest (one row per line)."""
    seen: set[tuple[str, str]] = set()
    scenes: list[tuple[str, str]] = []
    with Path(manifest_path).open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            key = (str(row["subject_id"]), str(row["scene_id"]))
            if key not in seen:
                seen.add(key)
                scenes.append(key)
    return scenes


def _posed_indices(
    object_pose_path: Path,
    frame_indices: set[int],
    *,
    confidence_threshold: float,
    cache: dict[Path, Mapping[str, object]],
) -> set[int]:
    if not object_pose_path.exists():
        return set()
    kept: set[int] = set()
    for frame_index in frame_indices:
        pose = load_object_pose_frame(object_pose_path, frame_index, cache=cache)
        if pose is not None and pose.confidence > confidence_threshold:
            kept.add(frame_index)
    return kept


def _wanted_indices(
    capture: cv2.VideoCapture,
    num_frames: int,
    config: ExtractConfig,
    object_pose_path: Path,
    pose_cache: dict[Path, Mapping[str, object]],
) -> set[int]:
    video_fps = capture.get(cv2.CAP_PROP_FPS) or DEFAULT_SOURCE_FPS
    wanted = set(sampled_frame_indices(num_frames, config.fps, video_fps=video_fps))
    if config.posed_only and wanted:
        wanted = _posed_indices(
            object_pose_path,
            wanted,
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
            cache=pose_cache,
        )
    return wanted


def _write_sampled(
    capture: cv2.VideoCapture,
    wanted: set[int],
    view_dir: Path,
    ext: str,
    encode_params: list[int],
) -> list[int]:
    """Sequentially decode; write the wanted frames; return the indices written."""
    written: list[int] = []
    last = max(wanted)
    index = 0
    # grab() advances cheaply; retrieve() only decodes the frames we keep.
    while index <= last:
        if not capture.grab():
            break
        if index in wanted:
            ok, frame = capture.retrieve()
            if ok:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                image_path = view_dir / f"{index:06d}{ext}"
                if not image_path.exists():
                    cv2.imwrite(str(image_path), gray, encode_params)
                written.append(index)
        index += 1
    return written


def extract_scene(
    root: str,
    out: str,
    subject_id: str,
    scene_id: str,
    config: ExtractConfig,
) -> list[dict[str, object]]:
    """Extract one recording's sampled frames; return its index rows."""
    root_path = Path(root)
    out_path = Path(out)
    paths = Show3DPaths(root_path, object_pose_version=config.object_pose_version)
    frame_ref = Show3DFrameRef(subject_id=subject_id, scene_id=scene_id, frame_index=0)
    ext, encode_params = _encode_params(config.image_format, config.quality)
    pose_cache: dict[Path, Mapping[str, object]] = {}
    rows: list[dict[str, object]] = []

    for view in config.views:
        video_path = paths.headset_path(frame_ref, int(view[-1]))
        if not video_path.exists():
            continue
        capture = cv2.VideoCapture(str(video_path))
        try:
            num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if num_frames <= 0:
                continue
            wanted = _wanted_indices(
                capture,
                num_frames,
                config,
                paths.object_pose_path(frame_ref),
                pose_cache,
            )
            if not wanted:
                continue
            view_dir = out_path / subject_id / scene_id / view
            view_dir.mkdir(parents=True, exist_ok=True)
            for index in _write_sampled(capture, wanted, view_dir, ext, encode_params):
                frame = Show3DFrameRef(
                    subject_id=subject_id, scene_id=scene_id, frame_index=index
                )
                rows.append(
                    {
                        "sample_id": frame.sample_id,
                        "subject_id": subject_id,
                        "scene_id": scene_id,
                        "frame_index": index,
                        "view": view,
                        "image": (view_dir / f"{index:06d}{ext}")
                        .relative_to(out_path)
                        .as_posix(),
                    }
                )
        finally:
            capture.release()
    return rows


def _select_scenes(
    root_path: Path,
    manifest: str | Path | None,
    subjects: Sequence[str] | None,
    require_object_pose: bool,
    object_pose_version: str,
    limit: int | None,
) -> list[tuple[str, str]]:
    """Resolve the ``(subject, scene)`` list to extract."""
    if manifest is not None:
        scenes = scenes_from_manifest(manifest)
    else:
        scenes = iter_scenes(root_path, set(subjects) if subjects else None)
        if require_object_pose:
            paths = Show3DPaths(root_path, object_pose_version=object_pose_version)
            scenes = [
                (subject, scene)
                for (subject, scene) in scenes
                if paths.object_pose_path(
                    Show3DFrameRef(subject_id=subject, scene_id=scene, frame_index=0)
                ).exists()
            ]
    if limit is not None:
        scenes = scenes[:limit]
    return scenes


def _run_scenes(
    args: list[tuple[str, str, str, str, ExtractConfig]],
    workers: int,
    verbose: bool,
) -> list[dict[str, object]]:
    """Extract every scene (parallel when workers > 1); return the index rows."""
    rows: list[dict[str, object]] = []
    if workers and workers > 1 and len(args) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(extract_scene, *a) for a in args]
            for done, future in enumerate(as_completed(futures), start=1):
                rows.extend(future.result())
                if verbose and done % 50 == 0:
                    print(f"  {done}/{len(args)} recordings")
    else:
        for done, a in enumerate(args, start=1):
            rows.extend(extract_scene(*a))
            if verbose and done % 50 == 0:
                print(f"  {done}/{len(args)} recordings")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def run_extraction(
    root: str | Path,
    out: str | Path,
    fps: int,
    *,
    manifest: str | Path | None = None,
    views: Sequence[str] = EGOCENTRIC_VIEWS,
    quality: int = 90,
    image_format: str = "jpeg",
    workers: int = 8,
    subjects: Sequence[str] | None = None,
    limit: int | None = None,
    posed_only: bool = False,
    require_object_pose: bool = False,
    object_pose_version: str = DEFAULT_OBJECT_POSE_VERSION,
    source_fps: float = DEFAULT_SOURCE_FPS,
    verbose: bool = False,
) -> dict[str, object]:
    """Extract frames under ``root`` and write the index.

    Pass ``manifest`` (a JSONL manifest with ``subject_id`` and ``scene_id`` per
    row) to extract exactly the scenes it names -- the reproducible way. Without
    it, the tool walks ``<root>/scenes``; ``require_object_pose`` keeps only the
    scenes with object-pose ground truth.
    """
    validate_fps(fps, source_fps)
    root_path = Path(root)
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)

    scenes = _select_scenes(
        root_path, manifest, subjects, require_object_pose, object_pose_version, limit
    )
    config = ExtractConfig(
        fps=fps,
        views=tuple(views),
        image_format=image_format,
        quality=quality,
        posed_only=posed_only,
        object_pose_version=object_pose_version,
    )
    args = [(str(root_path), str(out_path), s, sc, config) for (s, sc) in scenes]
    rows = _run_scenes(args, workers, verbose)

    index_path = out_path / "index.jsonl"
    _write_jsonl(index_path, rows)

    info: dict[str, object] = {
        "fps": fps,
        "views": list(views),
        "format": image_format,
        "quality": quality,
        "posed_only": posed_only,
        "manifest": str(manifest) if manifest is not None else None,
        "require_object_pose": require_object_pose,
        "num_recordings": len(scenes),
        "num_frames": len(rows),
    }
    (out_path / "extract_info.json").write_text(
        json.dumps(info, indent=2, sort_keys=True)
    )
    if verbose:
        print(
            f"Extracted {len(rows)} frames from {len(scenes)} recordings -> {out_path}"
        )
        print(f"  index: {index_path}")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract SHOW3D egocentric frames to images at a chosen fps"
    )
    parser.add_argument("--root", type=Path, required=True, help="SHOW3D mirror root")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest .jsonl naming the exact scenes to extract "
        "(the reproducible way; overrides tree walking)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        required=True,
        help="target sampling fps; REQUIRED (no default). Must be a whole number "
        "that divides the source rate: 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        default=list(EGOCENTRIC_VIEWS),
        choices=list(EGOCENTRIC_VIEWS),
        help="egocentric views to extract",
    )
    parser.add_argument(
        "--format",
        dest="image_format",
        default="jpeg",
        choices=["jpeg", "png"],
        help="output image format (jpeg is lossy but ~3-5x smaller)",
    )
    parser.add_argument(
        "--quality", type=int, default=90, help="JPEG quality (1-100; jpeg only)"
    )
    parser.add_argument("--workers", type=int, default=8, help="parallel recordings")
    parser.add_argument(
        "--subjects", nargs="+", default=None, help="restrict to these subject IDs"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only the first N recordings"
    )
    parser.add_argument(
        "--posed-only",
        action="store_true",
        help="skip frames without a confident object pose (needs object_pose/)",
    )
    parser.add_argument(
        "--require-object-pose",
        action="store_true",
        help="only scenes with object-pose ground truth (needs object_pose/)",
    )
    parser.add_argument("--object-pose-version", default=DEFAULT_OBJECT_POSE_VERSION)
    parser.add_argument(
        "--source-fps",
        type=float,
        default=DEFAULT_SOURCE_FPS,
        help="source capture fps used to validate --fps",
    )
    args = parser.parse_args()

    try:
        validate_fps(args.fps, args.source_fps)
    except ValueError as error:
        parser.error(str(error))

    run_extraction(
        args.root,
        args.out,
        args.fps,
        manifest=args.manifest,
        views=tuple(args.views),
        quality=args.quality,
        image_format=args.image_format,
        workers=args.workers,
        subjects=args.subjects,
        limit=args.limit,
        posed_only=args.posed_only,
        require_object_pose=args.require_object_pose,
        object_pose_version=args.object_pose_version,
        source_fps=args.source_fps,
        verbose=True,
    )


if __name__ == "__main__":
    main()

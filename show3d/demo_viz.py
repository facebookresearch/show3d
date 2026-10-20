# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Thin CLI for SHOW3D visualization; the library is :mod:`show3d.viz`.

    python -m show3d.demo_viz --out geometry.png
    python -m show3d.demo_viz --mode geometry --root DIR --manifest M.jsonl --out geom.png
    python -m show3d.demo_viz --mode overlay --root DIR --manifest M.jsonl --out overlay.png

With no --root/--manifest it renders a bundled synthetic scene. Modes: overlay
(skeleton + object on the frame) and geometry (3D skeleton + object). Overlay
draws on one camera (headset0, headset1, rig0 ... rig7), on both headsets side
by side with --view stereo, or on every rig camera in a grid with --view exo.

--model draws one hand model's posed meshes on a scene's frames instead, as an
image, or with --video as an MP4 of --num-frames frames:

    python -m show3d.demo_viz --root DIR --scene ISH822/aria_inspecting_3ab0 \
        --model mano --asset-dir mano_v1_2/models --frame 120 --out mano.png

--object-mesh draws the scene's object mesh, posed by its object pose, on one
frame (by default the first frame with an object pose). It draws the object's
color model once ``python -m show3d.hot3d_models`` has fetched it, else the
bundled mesh:

    python -m show3d.demo_viz --root DIR --scene PRI519/whiteboardmarker_writing-paper_2a1d \
        --object-mesh --out object_mesh.png
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Sequence
from pathlib import Path

from . import viz
from .dataset import CAMERA_VIEWS, DEFAULT_OBJECT_POSE_VERSION
from .hand_mesh import DEFAULT_MESH_HAND_POSE_VERSION, HAND_MODELS
from .hot3d_models import default_cache_dir
from .synthetic import build_synthetic_scene

DEFAULT_NUM_FRAMES: int = 300
# Flags that --object-mesh does not take.
_NOT_FOR_OBJECT_MESH: tuple[str, ...] = (
    "manifest",
    "mode",
    "asset_dir",
    "video",
    "num_frames",
    "hand_pose_version",
)


def _scene(
    parser: argparse.ArgumentParser, args: argparse.Namespace, flag: str
) -> tuple[str, str]:
    """``(subject, scene)`` from ``--scene``, for a mode that draws one view."""
    if args.root is None or args.scene is None or args.scene.count("/") != 1:
        parser.error(f"{flag} needs --root and --scene SUBJECT/SCENE")
    if args.view not in CAMERA_VIEWS:
        parser.error(f"{flag} draws one view; pick a camera, not stereo or exo")
    subject, scene = args.scene.split("/")
    return subject, scene


def _reject(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    flag: str,
    names: Sequence[str],
) -> None:
    """Stop with an error when any of the flags ``names`` was given with ``flag``."""
    given = [
        "--" + name.replace("_", "-")
        for name in names
        if getattr(args, name) not in (None, False)
    ]
    if given:
        parser.error(f"{flag} does not take {', '.join(given)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SHOW3D visualization demo")
    parser.add_argument("--root", type=Path, default=None, help="SHOW3D mirror root")
    parser.add_argument(
        "--manifest", type=Path, default=None, help="frame manifest .jsonl"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="output PNG (MP4 with --video)"
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=list(viz.MODES),
        help="what to draw (default: geometry)",
    )
    parser.add_argument(
        "--view",
        default="headset0",
        choices=[*CAMERA_VIEWS, viz.STEREO_VIEW, viz.EXO_VIEW],
        help="camera for overlay mode, --model and --object-mesh "
        "(stereo, exo: overlay only)",
    )
    meshes = parser.add_mutually_exclusive_group()
    meshes.add_argument(
        "--model",
        default=None,
        choices=list(HAND_MODELS),
        help="draw this hand model's meshes on --scene's frames",
    )
    meshes.add_argument(
        "--object-mesh",
        action="store_true",
        help="draw --scene's posed object mesh on one frame",
    )
    parser.add_argument(
        "--scene", default=None, help="SUBJECT/SCENE for --model and --object-mesh"
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=None,
        help="MANO model folder (mano) or MHR asset folder (mhr)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="first frame for --model (default: the first frame with a hand), or "
        "the frame for --object-mesh (default: the first frame with an object pose)",
    )
    parser.add_argument(
        "--video", action="store_true", help="with --model, write an MP4"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help=f"frames in the --video MP4 (default: {DEFAULT_NUM_FRAMES})",
    )
    parser.add_argument(
        "--hand-pose-version",
        default=None,
        help=f"hand_pose release for --model (default: {DEFAULT_MESH_HAND_POSE_VERSION})",
    )
    parser.add_argument(
        "--object-pose-version",
        default=DEFAULT_OBJECT_POSE_VERSION,
        help="object_pose release for the object pose: v1 or v2 "
        f"(default: {DEFAULT_OBJECT_POSE_VERSION})",
    )
    parser.add_argument(
        "--hot3d-models-dir",
        type=Path,
        default=None,
        help="with --object-mesh, where python -m show3d.hot3d_models put the color "
        f"models (default: {default_cache_dir()})",
    )
    args = parser.parse_args()

    if args.object_mesh:
        _reject(parser, args, "--object-mesh", _NOT_FOR_OBJECT_MESH)
        subject, scene = _scene(parser, args, "--object-mesh")
        viz.render_object_mesh(
            args.root,
            subject,
            scene,
            args.out or Path("object_mesh.png"),
            view=args.view,
            frame_index=args.frame,
            object_pose_version=args.object_pose_version,
            hot3d_models_dir=args.hot3d_models_dir,
            verbose=True,
        )
        return
    if args.hot3d_models_dir is not None:
        parser.error("--hot3d-models-dir goes with --object-mesh")

    if args.model is not None:
        subject, scene = _scene(parser, args, "--model")
        default_out = Path(f"{args.model}.mp4" if args.video else f"{args.model}.png")
        num_frames = DEFAULT_NUM_FRAMES if args.num_frames is None else args.num_frames
        viz.render_hand_meshes(
            args.root,
            subject,
            scene,
            args.model,
            args.out or default_out,
            view=args.view,
            frame_index=args.frame,
            num_frames=num_frames if args.video else None,
            asset_dir=args.asset_dir,
            hand_pose_version=args.hand_pose_version or DEFAULT_MESH_HAND_POSE_VERSION,
            verbose=True,
        )
        return

    out = args.out or Path("viz.png")
    mode = args.mode or "geometry"
    print(f"SHOW3D visualization demo (mode={mode})\n")
    if args.root is not None and args.manifest is not None:
        viz.run_visualization(
            args.root,
            args.manifest,
            out,
            mode=mode,
            view=args.view,
            object_pose_version=args.object_pose_version,
            verbose=True,
        )
        return

    with tempfile.TemporaryDirectory() as td:
        manifest_path = build_synthetic_scene(Path(td))
        print("No --root/--manifest given; visualizing a synthetic toy scene.")
        viz.run_visualization(
            Path(td),
            manifest_path,
            out,
            mode=mode,
            view=args.view,
            verbose=True,
        )


if __name__ == "__main__":
    main()

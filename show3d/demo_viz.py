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
(skeleton + object on the frame) and geometry (3D skeleton + object). With
--view stereo, overlay draws headset0 and headset1 side by side.

--model draws one hand model's posed meshes on a scene's frames instead, as an
image, or with --video as an MP4 of --num-frames frames:

    python -m show3d.demo_viz --root DIR --scene ISH822/aria_inspecting_3ab0 \
        --model mano --asset-dir mano_v1_2/models --frame 120 --out mano.png
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from . import viz
from .dataset import EGOCENTRIC_VIEWS
from .hand_mesh import DEFAULT_MESH_HAND_POSE_VERSION, HAND_MODELS
from .synthetic import build_synthetic_scene


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
        "--mode", default="geometry", choices=list(viz.MODES), help="what to draw"
    )
    parser.add_argument(
        "--view",
        default="headset0",
        choices=[*EGOCENTRIC_VIEWS, viz.STEREO_VIEW],
        help="egocentric view for overlay mode and --model (stereo: overlay only)",
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=list(HAND_MODELS),
        help="draw this hand model's meshes on --scene's frames",
    )
    parser.add_argument("--scene", default=None, help="SUBJECT/SCENE for --model")
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
        help="first frame for --model (default: the first frame with a hand)",
    )
    parser.add_argument(
        "--video", action="store_true", help="with --model, write an MP4"
    )
    parser.add_argument(
        "--num-frames", type=int, default=300, help="frames in the --video MP4"
    )
    parser.add_argument(
        "--hand-pose-version",
        default=DEFAULT_MESH_HAND_POSE_VERSION,
        help="hand_pose release for --model",
    )
    args = parser.parse_args()

    if args.model is not None:
        if args.root is None or args.scene is None or args.scene.count("/") != 1:
            parser.error("--model needs --root and --scene SUBJECT/SCENE")
        if args.view == viz.STEREO_VIEW:
            parser.error("--model draws one view; pick headset0 or headset1")
        subject, scene = args.scene.split("/")
        default_out = Path(f"{args.model}.mp4" if args.video else f"{args.model}.png")
        viz.render_hand_meshes(
            args.root,
            subject,
            scene,
            args.model,
            args.out or default_out,
            view=args.view,
            frame_index=args.frame,
            num_frames=args.num_frames if args.video else None,
            asset_dir=args.asset_dir,
            hand_pose_version=args.hand_pose_version,
            verbose=True,
        )
        return

    out = args.out or Path("viz.png")
    print(f"SHOW3D visualization demo (mode={args.mode})\n")
    if args.root is not None and args.manifest is not None:
        viz.run_visualization(
            args.root,
            args.manifest,
            out,
            mode=args.mode,
            view=args.view,
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
            mode=args.mode,
            view=args.view,
            verbose=True,
        )


if __name__ == "__main__":
    main()

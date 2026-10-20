# SHOW3D

Python tools for [SHOW3D](https://huggingface.co/datasets/facebook/show3d-dataset),
a dataset of 3D hands and objects captured in the wild (CVPR 2026).

[Paper](https://arxiv.org/abs/2603.28760) |
[Dataset](https://huggingface.co/datasets/facebook/show3d-dataset)

![Both egocentric views of an outdoor SHOW3D frame, with the hand skeletons and the vase drawn on them](docs/overlay.png)

SHOW3D has 2,137 recordings of hands using objects in the wild, 20 hours in
total. Each recording has two egocentric views from a Meta Quest 3 headset at 60
fps and up to eight exocentric views from a camera rig worn on the back. The
labels are marker-less 3D hand poses, object poses and interaction fields.

With this package you can:

- load frames: video paths, camera calibration, hand poses and object poses
- draw hands and objects on a frame, or in 3D
- pose and draw hand meshes in three hand models: UmeTrack, MANO and MHR
- draw object meshes on a frame, posed by the object poses
- extract frames to images for fast training

[`show3d/README.md`](show3d/README.md) describes the modules, the options and
the tests.

## Install

```bash
git clone https://github.com/facebookresearch/show3d.git
cd show3d
pip install -r requirements.txt
```

The core needs only numpy and OpenCV, and drawing needs matplotlib. Hand meshes
need a few more packages; see [Render hand meshes](#render-hand-meshes). To
check the install without the data, draw a small synthetic scene:

```bash
python -m show3d.demo_viz --out geometry.png
```

## Get the data

Download SHOW3D from Hugging Face:
[`facebook/show3d-dataset`](https://huggingface.co/datasets/facebook/show3d-dataset).
The code expects the released layout under one root folder:

```
<root>/
├── scenes/<subject>/<scene>/
│   ├── headset0.mp4, headset1.mp4              # egocentric videos
│   ├── rig0.mp4 ... rig7.mp4                   # exocentric videos
│   ├── camera_calibration/headset{0,1}.json   # intrinsics + per-frame pose
│   ├── camera_calibration/rig{0..7}.json      # intrinsics + one static pose
│   └── metadata/frame_info.json
├── object_pose/
│   ├── <version>/scenes/<subject>/<scene>/object_pose.json
│   └── <version>/test_scenes/<subject>/<scene>/  # v2 on: the test split, same file
├── hand_pose/
│   ├── <version>/scenes/<subject>/<scene>/hand_pose_umetrack.json   # v1, v2: hand_pose.json
│   ├── <version>/test_scenes/<subject>/<scene>/  # v3 on: the test split, same files
│   └── hand_profiles/<subject>/profile_umetrack.json
└── interaction_field/
    ├── v1/scenes/<subject>/<scene>/interaction_field.jsonl
    └── v1/test_scenes/<subject>/<scene>/interaction_field.jsonl  # the test split
```

From hand_pose v3 on, each scene also has `hand_pose_mano.json` and
`hand_pose_mhr.json`, and each subject has `profile_mano.json` and
`profile_mhr.json`. The loaders find a test subject's files in `test_scenes/`,
and `TEST_SUBJECTS` in `show3d.dataset` lists the test subjects.

## Load frames

`Show3DDataset` takes a root folder and a list of frames, and each item holds
one frame's video paths, calibration, hand poses and object pose. `posed_frames`
lists the frames of a recording that have an object pose:

```python
from show3d.dataset import posed_frames, Show3DDataset

root = "/path/to/show3d"
frames = posed_frames(root, "ISH822", "aria_inspecting_3ab0")[::30]
dataset = Show3DDataset(root, frames)

frame = dataset[0]
frame.left_hand           # confidence and (21, 3) landmarks_world_mm, or None
frame.object_pose         # confidence, rotation and translation_mm, or None
frame.views["headset0"]   # video_path and calibration (intrinsics, world-from-camera)
```

The loader reads object_pose v2 and hand_pose v2 by default; pass
`hand_pose_version="v3"` for the test split's hand poses. It loads the two
headset cameras unless you pass `views=CAMERA_VIEWS` for all ten. Positions are
in millimeters, in the frame of the back-mounted rig, which moves with the
person. Use a frame's headset pose only when `frame.headset_pose_valid` is true.

## Visualize a frame

`show3d.demo_viz` draws one frame of a manifest, written for example from
`posed_frames`:

```bash
python -m show3d.demo_viz --root /path/to/show3d --manifest frames.jsonl \
    --mode overlay --view stereo --out overlay.png
python -m show3d.demo_viz --root /path/to/show3d --manifest frames.jsonl \
    --mode overlay --view exo --out exo.png
python -m show3d.demo_viz --root /path/to/show3d --manifest frames.jsonl \
    --mode geometry --out geometry.png
```

The `overlay` mode projects the hands and the object onto one camera
(`--view headset0` to `rig7`), both headsets (`stereo`, as in the image at the
top) or every rig camera (`exo`):

![The eight rig views of the same frame, with the hand skeletons and the vase drawn on them](docs/exo.png)

The `geometry` mode draws them in 3D, seen from near the headset0 camera. Left
hands are orange and right hands cyan in every render:

![The hand skeletons and the vase of the same frame in 3D](docs/geometry.png)

## Render hand meshes

From hand_pose v3 on, every hand comes in three hand models: UmeTrack (the
native solve), MANO and MHR. `demo_viz --model` draws the meshes on one frame, or
on a frame range with `--video`:

```bash
python -m show3d.demo_viz --root /path/to/show3d --scene ISH822/aria_inspecting_3ab0 \
    --model mano --asset-dir /path/to/mano/models --frame 600 --out mano.png
python -m show3d.demo_viz --root /path/to/show3d --scene ISH822/aria_inspecting_3ab0 \
    --model mhr --asset-dir /path/to/mhr/assets --video --num-frames 300 --out mhr.mp4
```

![The hand meshes of one frame in MANO and MHR](docs/hand_mesh.png)

Each model needs its own packages and files:

- **UmeTrack:** `pip install torch projectaria-tools`, with the `hot3d/` folder of
  [HOT3D](https://github.com/facebookresearch/hot3d) on `PYTHONPATH`.
- **MANO:** `pip install torch smplx`, with `MANO_LEFT.pkl` and `MANO_RIGHT.pkl`
  from [MANO](https://mano.is.tue.mpg.de) in `--asset-dir`, cleaned of chumpy
  objects by smplx's `tools/clean_ch.py`.
- **MHR:** `pip install torch mhr pymomentum-cpu` (Python 3.12 or 3.13), with the
  asset folder of [MHR](https://github.com/facebookresearch/MHR) as `--asset-dir`.

From Python, `HandMeshScene` returns world-frame vertices in mm and their faces:

```python
from show3d.hand_mesh import HandMeshScene, RIGHT_SLOT

scene = HandMeshScene("/path/to/show3d", "ISH822", "aria_inspecting_3ab0", "mano",
                      asset_dir="/path/to/mano/models")
mesh = scene.mesh(600, RIGHT_SLOT)  # None when the hand has no confident pose
```

## Render object meshes

The 22 objects that SHOW3D shares with [HOT3D](https://github.com/facebookresearch/hot3d)
come with their meshes in `show3d/assets/objects/`, in millimeters and in the
object frame of the object poses. `python -m show3d.hot3d_models` downloads their
color models once. `demo_viz --object-mesh` draws a scene's object mesh, posed by
its object pose, with the color model when it has it:

```bash
python -m show3d.demo_viz --root /path/to/show3d --scene SCH724/waffles_tossing-between-hands_e22d \
    --object-mesh --frame 1201 --out waffles.png
```

![The waffles box drawn on the same frame with its color model and with the bundled mesh](docs/object_mesh.png)

The same from Python:

```python
from show3d.dataset import load_object_mesh_with_faces, posed_frames, Show3DDataset
from show3d.hot3d_models import load_color_mesh
from show3d.viz import draw_object_mesh

root = "/path/to/show3d"
frames = posed_frames(root, "PRI519", "whiteboardmarker_writing-paper_2a1d")
frame = Show3DDataset(root, frames[:1], decode_images=True)[0]
view = frame.views["headset0"]
mesh = load_color_mesh("whiteboardmarker") or load_object_mesh_with_faces("whiteboardmarker")
image = draw_object_mesh(view.image, mesh, frame.object_pose, view.calibration)
```

## Extract frames for training

`show3d.extract_images` decodes each video once and writes the frames you keep as
images, which trains much faster than seeking in the MP4s:

```bash
python -m show3d.extract_images --root /path/to/show3d --out frames/ --fps 10 --posed-only
```

`--posed-only` keeps the frames with a confident object pose, and `--views` picks
the cameras. The output has `index.jsonl` with one row per image, and each row's
`sample_id` (`SUBJECT/SCENE:FRAME`) joins per-frame labels to the images.

## Interaction field

Interaction field v1 labels give, on each labeled frame, the vector in mm from
each of a hand's 21 landmarks to the nearest vertex of the object mesh. Each
recording has one `interaction_field.jsonl`, one line per labeled frame, keyed by
the `sample_id` that `extract_images` writes:

```python
import json

path = f"{root}/interaction_field/v1/scenes/{subject}/{scene}/interaction_field.jsonl"
with open(path) as f:
    for line in f:
        row = json.loads(line)  # sample_id, left_to_object, right_to_object
        # each hand: 21 vectors [x, y, z] in mm, or None when the hand has no label
```

v1 comes from hand pose v3 and object pose v2 and covers both splits;
`interaction_field/README.md` in the dataset gives the details. These labels
differ from those of the
[Interaction Field Estimation Challenge](https://github.com/patrickqrim/SHOW3D-dataset-api/tree/main/show3d/interaction_field)
at the HANDS workshop, ECCV 2026, which were computed on older versions of
hand/object poses.

## License

The code is released under Creative Commons Attribution-NonCommercial 4.0
International (CC BY-NC 4.0); see `LICENSE`.

The object meshes are HOT3D object models: the ones bundled in
`show3d/assets/objects/` (BOP-HOT3D `models_eval`) and the color models that
`show3d.hot3d_models` downloads (BOP-HOT3D `models`). They come under the terms
of the [HOT3D Dataset License Agreement](https://www.projectaria.com/datasets/hot3d/license/)
(also in [`bop-benchmark/hot3d`](https://huggingface.co/datasets/bop-benchmark/hot3d/blob/main/hot3d_dataset_license_agreement.pdf)),
which puts them under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
with one change: you may not sell them or put them into a product to be sold.
See `show3d/assets/objects/ATTRIBUTION.md`.

## Citation

If you use SHOW3D, please cite:

```bibtex
@InProceedings{Rim_2026_CVPR,
    author    = {Rim, Patrick and Harris, Kevin and Copple, Braden and Han, Shangchen and Xie, Xu and Shugurov, Ivan and An, Sizhe and Wen, He and Wong, Alex and Hodan, Tomas and He, Kun},
    title     = {SHOW3D: Capturing Scenes of 3D Hands and Objects in the Wild},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {7111-7120}
}
```

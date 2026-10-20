# The `show3d` package

This page describes the modules and the options behind the examples in the
[top-level README](../README.md).

## Modules

```
show3d/
├── dataset.py            # dataset loader, paths and pose parsing
├── camera.py             # pinhole projection
├── viz.py                # drawing: skeletons, objects, hand and object meshes
├── demo_viz.py           # visualization command line
├── hand_mesh.py          # posed hand meshes; one hand_mesh_<model>.py per model
├── extract_images.py     # extract frames to images at a chosen fps
├── hot3d_models.py       # fetch and load the HOT3D color models
├── synthetic.py          # a tiny synthetic scene for demos and tests
├── assets/objects/       # object meshes (.glb, from HOT3D)
└── tests/                # unit tests
```

Run the tests from the repository root:

```bash
python -m unittest discover
```

## Frames and versions

- **Versions.** `Show3DPaths` and `Show3DDataset` read object_pose v2 and
  hand_pose v2 by default. Pass `hand_pose_version` or `object_pose_version` to
  pick another release. The test split's hand poses are in
  `hand_pose/<version>/test_scenes/` from v3 on, and its object poses in
  `object_pose/<version>/test_scenes/` from v2 on; the loaders pick the folder
  from the subject ID.
- **Object pose frames.** A v1 file has an entry on every video frame. A v2 file
  has an entry on every other video frame, from frame 0 or from frame 1, or on
  every video frame, and in a few v2 files the grid shifts by one frame at one
  or more points. A fixed stride from frame 0 can therefore miss all of a
  recording's poses. `posed_frames` reads the file's keys and keeps the frames
  whose entry has a pose; `min_confidence` keeps only the poses above that
  confidence.
- **Frame lists.** List frames as `Show3DFrameRef(subject_id, scene_id,
  frame_index)`, or keep them in a manifest: a JSONL file with one
  `{"subject_id", "scene_id", "frame_index"}` object per line. Load it with
  `Show3DDataset.from_manifest_jsonl(root, path)`, and write one with
  `write_frame_manifest_jsonl`.
- **Images.** The loader returns paths only; `decode_images=True` also decodes
  each frame with OpenCV.
- **Scene IDs** have the form `<object>_<action>_<hash>`, for example
  `bbq_cleaning-the-outside_277b`; the part before the first underscore names
  the object.

## Cameras

`views=CAMERA_VIEWS` (from `show3d.dataset`) loads all ten cameras and
`views=EXOCENTRIC_VIEWS` the eight rig cameras `rig0` to `rig7`. A rig camera has
one fixed pose in the rig frame, so it is valid on every frame. A scene can lack
some rig cameras, and `frame.views[name].missing` is true when
`metadata/frame_info.json` lists that camera as missing on the frame.

Positions are in millimeters, in the world frame of the scene's calibration.
This world frame is the frame of the back-mounted rig, so it moves with the
person; the dataset card gives the details.

### Headset pose validity

Use a frame's headset pose only when `frame.headset_pose_valid` is true.

- **Calibration files with `pose_contract_version` 1** carry `is_pose_valid` and
  `pose_source` (`mocap`, `vio`, `endpoint_interpolation`,
  `smooth_mocap_interpolation` or `legacy_unspecified`) per frame. The loader
  returns `t_world_from_camera` only when `is_pose_valid` is true.
- **Older calibration files** have only `is_synthesized`, which is true when
  interpolation produced the pose. The loader returns `t_world_from_camera` only
  when `is_synthesized` is false.

Pin the data and code revisions together to keep the same set of valid frames.

## Visualization

`demo_viz` picks the first frame of the manifest with a camera pose in every
view it draws and a confident hand. The `geometry` mode draws in the headset0
camera's frame, so it lines up with the headset image. The object comes from
object_pose v2, and `--object-pose-version v1` draws v1. The drawing functions
live in `show3d.viz` and the camera projection in `show3d.camera`.

## Hand meshes

`show3d.hand_mesh` poses each hand model in the scene's world frame.
`demo_viz --model` draws on one camera's frame (`--view`, headset0 by default).
UmeTrack uses HOT3D's UmeTrack loader and also works with hand_pose v2
(`--hand-pose-version v2`). For MANO, smplx loads `MANO_LEFT.pkl` and
`MANO_RIGHT.pkl` without chumpy only after `tools/clean_ch.py` removes their
chumpy objects, which runs under Python 2 with chumpy installed.

`HandMeshScene.mesh` returns None when the hand is missing, when its confidence
is 0.1 (the accept gate) or lower, or, for UmeTrack, when the frame has no world
geometry. Pass `confidence_threshold=0.5` for the high-confidence gate.

## Object meshes

- **Bundled meshes.** `show3d/assets/objects/` holds the BOP-HOT3D `models_eval`
  models of the 22 objects, with no color; `HOT3D_OBJECT_LID` in
  `show3d.dataset` lists them.
- **Color models.** `python -m show3d.hot3d_models` downloads BOP-HOT3D's
  `hot3d_models.zip` (137 MB) from
  [Hugging Face](https://huggingface.co/datasets/bop-benchmark/hot3d) at a pinned
  revision, checks its sha256 and keeps the 22 objects' models (44 files,
  174 MB) in `~/.cache/show3d/hot3d_models`, or under `$XDG_CACHE_HOME` when it
  is set. Nothing is downloaded until you run it, and a second run that finds
  all 44 files downloads nothing. `--zip` extracts a copy of the zip you already
  have.
- **Drawing.** `demo_viz --object-mesh` draws the color model when it is in the
  cache and the bundled mesh in magenta otherwise, so it also works offline;
  when it falls back, it prints the cache path it searched and the command that
  fetches the color model. Either way it outlines the silhouette in green.
  Without `--frame` it draws the scene's first frame with an object pose, and
  `--hot3d-models-dir` points it at another cache folder.
- **Loaders.** `load_color_mesh` returns `vertices_mm`, `faces` and
  `face_colors`, and `load_object_mesh_with_faces` returns the bundled mesh's
  `vertices_mm` and `faces`. Both return None for the five objects that HOT3D
  does not have: keyboard2, cancoke, windex, clock and mug3. `load_color_mesh`
  also returns None until the model is in the cache. The largest color model
  takes a few seconds to read, so load each model once.

## Frame extraction

`show3d.extract_images` options:

- `--fps` is required. It must divide the 60 fps source evenly: 1, 2, 3, 4, 5, 6,
  10, 12, 15, 20, 30 or 60.
- `--manifest` limits the run to the scenes a manifest names. Without it,
  `--require-object-pose` keeps only the scenes with object poses. From
  object_pose v2 on, those include the test subjects' scenes; `--subjects`
  restricts the run.
- `--posed-only` keeps only the frames with an object pose above confidence 0.5,
  sampled on the frames that each recording's object_pose file labels: it keeps
  a labeled frame once it is the stride (60 / `--fps`) past the last one kept.
  For object_pose v1, which labels every frame, that is every stride-th frame
  from frame 0. A v2 file that labels every other frame, starting at frame 1,
  still gets its frames. On such a file an odd stride grows by one frame, so the
  rate rounds down to the grid: `--fps 20` gives 15 fps and `--fps 12` gives 10.
  A v2 run with such an `--fps` prints a one-line note. Without `--posed-only`,
  frames count from frame 0, so they can miss every label of such a v2 file.
- `--views headset0`, `--format png`, `--quality 80` and `--workers 16` set the
  cameras, image format, JPEG quality and parallelism. `--views` takes any of the
  ten cameras, for example `--views rig0 rig1`; the default is the two headsets.

The output has one image per frame and view, plus `index.jsonl` with one row per
image. Each row's `sample_id` (`SUBJECT/SCENE:FRAME`) is the same for every view
of a frame.

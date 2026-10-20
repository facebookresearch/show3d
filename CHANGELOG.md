# Changelog

## 2026-10-19

- The README is shorter; `show3d/README.md` holds the module layout, the
  frame and camera details and the full options.
- The README's data layout lists `interaction_field/v1`, the hand-to-object
  vectors released on Hugging Face on October 19, 2026. They differ from the
  labels of the ECCV 2026 Interaction Field Estimation Challenge.
- object_pose v2 is the default: `DEFAULT_OBJECT_POSE_VERSION` is `"v2"`, so
  the loader, `posed_frames`, `extract_images` and `demo_viz` read
  `object_pose/v2/` unless given another version. Pass
  `object_pose_version="v1"` or `--object-pose-version v1` for v1.
- Test-split object pose: `Show3DPaths.object_pose_path` reads the test
  subjects' object pose from `object_pose/<version>/test_scenes/` from v2 on;
  v1 keeps every subject in `scenes/`. `object_pose_subject_dir` returns a
  subject's folder. `extract_images --require-object-pose` therefore keeps the
  test subjects' scenes too; `--subjects` restricts the run, and
  `TEST_SUBJECTS` lists the test subjects.
- Test-split hand pose: `Show3DPaths.hand_pose_path` and `HandMeshScene` read
  the 6 test subjects' hand pose from `hand_pose/<version>/test_scenes/`, which
  ships from hand_pose v3 on. `hand_pose_scene_dir` returns a recording's
  folder, and `TEST_SUBJECTS` lists the test subjects.
- Object meshes on a frame: `demo_viz --object-mesh` and `viz.render_object_mesh`
  draw a scene's object mesh, posed by its object pose, on any of the ten
  cameras, and `viz.draw_object_mesh` draws it on any image of the frame, with
  the silhouette outlined.
- Color models: `python -m show3d.hot3d_models` downloads BOP-HOT3D's
  `hot3d_models.zip` at a pinned revision, checks its sha256 and keeps the 22
  objects' textured models. The object mesh renders use them when present and
  the bundled meshes otherwise. `load_color_mesh` returns one with a color per
  face.
- `load_object_mesh_with_faces` returns a bundled mesh with its triangles, and
  `HOT3D_BOP_OBJECT_ID` maps each alias to its BOP-HOT3D model id.
- The whiteboard marker mesh (`whiteboardmarker.glb`, BOP-HOT3D `obj_000032`)
  joins the bundled meshes, which now cover all 22 HOT3D objects.
- `posed_frames` and `object_pose_frame_indices` list the frames that a
  recording's object_pose file labels, read from its keys, so they follow each
  file's own frame grid.
- `extract_images --posed-only` samples each recording on the frames its
  object_pose file labels. It used to count from frame 0, which at an even
  stride kept no frame of an object_pose v2 file that labels odd frames.
- `demo_viz --object-pose-version` and `run_visualization(object_pose_version=)`
  pick the object_pose release of the overlay and geometry renders.
- `HOT3D_OBJECT_LID` maps brushholder to LID 16; it was 15.

## 2026-10-04

First release in facebookresearch/show3d. The dataset API used to live in
[patrickqrim/SHOW3D-dataset-api](https://github.com/patrickqrim/SHOW3D-dataset-api),
which keeps the Interaction Field Estimation Challenge kit.

- Dataset loader for the egocentric views: video paths, calibration, hand poses
  and object poses.
- Overlay and 3D renders of hands and objects.
- Posed hand meshes in UmeTrack, MANO and MHR (hand_pose v3).
- Frame extraction to images for training.

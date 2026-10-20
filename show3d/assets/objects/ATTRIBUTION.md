# Bundled object meshes — attribution

`<alias>.glb` — one GLB mesh per bundled object (object frame, **millimeters**),
keyed by SHOW3D object alias. Pose a mesh per frame with the released `R,t`
(`ObjectPoseFrame.pose_vertices`) to get the object surface in world space.

The 22 meshes are HOT3D object models from BOP-HOT3D
(https://huggingface.co/datasets/bop-benchmark/hot3d): the `models_eval/` folder
of `hot3d_models.zip`, converted from PLY to GLB in millimeters with the geometry
unchanged. `show3d.dataset.HOT3D_BOP_OBJECT_ID` maps each alias to its BOP model
id. `show3d.hot3d_models` downloads the full color models (`models/` of the same
zip, PLY with PNG textures) on request; this repository does not include them.

Both sets of meshes come under the terms of the HOT3D Dataset License Agreement
(https://www.projectaria.com/datasets/hot3d/license/, also
https://huggingface.co/datasets/bop-benchmark/hot3d/blob/main/hot3d_dataset_license_agreement.pdf).
It licenses the HOT3D 3D object models under Creative Commons
Attribution-ShareAlike 4.0 International (CC BY-SA 4.0,
https://creativecommons.org/licenses/by-sa/4.0/), with the modification that you
may not sell them or incorporate them into a product to be sold. The code in this
repository keeps its own license (`LICENSE`).

HOT3D: https://github.com/facebookresearch/hot3d

```bibtex
@article{banerjee2024hot3d,
  title={{HOT3D}: Hand and Object Tracking in 3D from Egocentric Multi-View Videos},
  author={Banerjee, Prithviraj and others},
  journal={arXiv preprint arXiv:2411.19167},
  year={2024}
}
```

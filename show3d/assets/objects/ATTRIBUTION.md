# Bundled object meshes — attribution

`<alias>.glb` — one GLB mesh per bundled object (object frame, **millimeters**),
keyed by SHOW3D object alias. Pose a mesh per frame with the released `R,t`
(`ObjectPoseFrame.pose_vertices`) to get the object surface in world space.

The meshes are the HOT3D object models from the BOP HOT3D release
(https://huggingface.co/datasets/bop-benchmark/hot3d, `object_models_eval`),
redistributed here under Creative Commons Attribution-NonCommercial 4.0
International (CC BY-NC 4.0), consistent with the SHOW3D and HOT3D licenses.

HOT3D: https://github.com/facebookresearch/hot3d

```bibtex
@article{banerjee2024hot3d,
  title={{HOT3D}: Hand and Object Tracking in 3D from Egocentric Multi-View Videos},
  author={Banerjee, Prithviraj and others},
  journal={arXiv preprint arXiv:2411.19167},
  year={2024}
}
```

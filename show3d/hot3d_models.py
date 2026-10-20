# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fetch and load the color models of the 22 HOT3D objects.

The bundled meshes (``assets/objects/<alias>.glb``) are BOP-HOT3D's
``models_eval`` models, which have no color. The same ``hot3d_models.zip`` on
Hugging Face (``bop-benchmark/hot3d``) also holds the full models: one ASCII PLY
per object with a PNG texture, in millimeters and in the object frame of the
object poses. :func:`fetch_hot3d_models` downloads that zip (137 MB) at a pinned
revision, checks its sha256 and extracts the 22 objects' 44 PLY and PNG files
(174 MB) into a cache folder. Nothing is downloaded until you run it, and a run
that finds all 44 files already in the cache downloads nothing:

    python -m show3d.hot3d_models
    python -m show3d.hot3d_models --zip /path/to/hot3d_models.zip --cache-dir DIR

:func:`load_color_mesh` then returns a fetched model with a color per face. The
models come under the HOT3D Dataset License Agreement; see
``assets/objects/ATTRIBUTION.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

from .dataset import HOT3D_BOP_OBJECT_ID, ObjectMesh

HOT3D_MODELS_REVISION: str = "30fe9674782f32e1e5edba98476b6ff4300132c5"
HOT3D_MODELS_URL: str = (
    "https://huggingface.co/datasets/bop-benchmark/hot3d/resolve/"
    f"{HOT3D_MODELS_REVISION}/hot3d_models.zip"
)
HOT3D_MODELS_SHA256: str = (
    "7b004ed35bcef4242ac50040cfb44140c37de4e37ce023a00affcf0f20a58f64"
)
# sha256 of the sha256sum listing of the 44 extracted files in name order, i.e.
# ``sha256sum obj_*.ply obj_*.png | sort -k2 | sha256sum`` in the cache folder.
HOT3D_MODELS_FILES_SHA256: str = (
    "52ee0033215182924aa3f60a1b0a8527c0fbfcbdb86831c845479b2397b1bb50"
)
_CHUNK_BYTES: int = 1 << 20
# Seconds that the download may wait on the connection before it gives up.
_DOWNLOAD_TIMEOUT_S: float = 60.0


def default_cache_dir() -> Path:
    """``$XDG_CACHE_HOME/show3d/hot3d_models``, by default
    ``~/.cache/show3d/hot3d_models``."""
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "show3d" / "hot3d_models"


def _model_name(bop_id: int) -> str:
    return f"obj_{bop_id:06d}"


def _model_files() -> list[str]:
    """The 44 file names that :func:`fetch_hot3d_models` extracts, in name order."""
    return sorted(
        _model_name(bop_id) + suffix
        for bop_id in set(HOT3D_BOP_OBJECT_ID.values())
        for suffix in (".ply", ".png")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_cached(out: Path) -> bool:
    """Whether ``out`` holds all 44 extracted files with their published bytes."""
    names = _model_files()
    if not all((out / name).is_file() for name in names):
        return False
    listing = "".join(f"{_sha256(out / name)}  {name}\n" for name in names)
    return hashlib.sha256(listing.encode()).hexdigest() == HOT3D_MODELS_FILES_SHA256


def _extract(zip_path: Path, out: Path) -> None:
    """Check the zip's sha256, then copy each object's PLY and PNG out of its
    ``models/`` folder; a file appears under its final name only once complete."""
    digest = _sha256(zip_path)
    if digest != HOT3D_MODELS_SHA256:
        raise ValueError(
            f"{zip_path} has sha256 {digest}, not the {HOT3D_MODELS_SHA256} of "
            f"hot3d_models.zip at revision {HOT3D_MODELS_REVISION}"
        )
    with zipfile.ZipFile(zip_path) as archive:
        for bop_id in sorted(set(HOT3D_BOP_OBJECT_ID.values())):
            # The PLY last, so that it implies its texture.
            for suffix in (".png", ".ply"):
                name = _model_name(bop_id) + suffix
                partial = out / f"{name}.part"
                try:
                    with (
                        archive.open(f"models/{name}") as src,
                        partial.open("wb") as dst,
                    ):
                        shutil.copyfileobj(src, dst, _CHUNK_BYTES)
                    os.replace(partial, out / name)
                finally:
                    partial.unlink(missing_ok=True)


def _download(out: Path, verbose: bool) -> None:
    """Download the zip into ``out``, extract it and delete it, also on failure."""
    downloaded = out / "hot3d_models.zip.part"
    if verbose:
        print(f"Downloading {HOT3D_MODELS_URL} (137 MB)")
    try:
        with (
            urllib.request.urlopen(
                HOT3D_MODELS_URL, timeout=_DOWNLOAD_TIMEOUT_S
            ) as src,
            downloaded.open("wb") as dst,
        ):
            shutil.copyfileobj(src, dst, _CHUNK_BYTES)
        _extract(downloaded, out)
    finally:
        downloaded.unlink(missing_ok=True)


def fetch_hot3d_models(
    cache_dir: str | Path | None = None,
    *,
    zip_path: str | Path | None = None,
    verbose: bool = False,
) -> Path:
    """Put the 22 objects' color models, 44 files of 174 MB, into ``cache_dir``
    (default :func:`default_cache_dir`) and return it.

    When the cache already holds the 44 files with the published bytes, nothing
    is downloaded or extracted. Otherwise the models come from ``zip_path`` when
    given, else from a download of ``HOT3D_MODELS_URL`` (137 MB) that is deleted
    afterwards. Either way the zip must have the sha256 ``HOT3D_MODELS_SHA256``.
    A run that fails leaves no partial file behind.
    """
    out = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    num_objects = len(HOT3D_BOP_OBJECT_ID)
    if _is_cached(out):
        if verbose:
            print(f"The color models of {num_objects} objects are already in {out}")
        return out
    out.mkdir(parents=True, exist_ok=True)
    if zip_path is not None:
        _extract(Path(zip_path), out)
    else:
        _download(out, verbose)
    if verbose:
        print(f"Extracted the color models of {num_objects} objects to {out}")
    return out


def _ply_layout(header: list[str]) -> tuple[str, int, list[str], int, list[str]]:
    """``(texture file, vertex count, vertex properties, face count, face
    properties)`` of an ASCII PLY header with one texture, a vertex element and a
    face element whose one list property is ``vertex_indices``."""
    texture: str | None = None
    elements: list[tuple[str, int, list[str]]] = []
    lists: list[str] = []
    for line in header:
        words = line.split()
        if words[:1] == ["format"] and words[1:2] != ["ascii"]:
            raise ValueError(f"only ASCII PLY files are supported, got {line!r}")
        if words[:2] == ["comment", "TextureFile"]:
            texture = words[2]
        elif words[:1] == ["element"]:
            elements.append((words[1], int(words[2]), []))
        elif words[:1] == ["property"]:
            elements[-1][2].append(words[-1])
            if words[1] == "list":
                lists.append(f"{elements[-1][0]} {words[-1]}")
    names = [name for name, _count, _properties in elements]
    if texture is None or names != ["vertex", "face"]:
        raise ValueError(
            f"expected a textured PLY with vertices and faces, got {names}"
        )
    if lists != ["face vertex_indices"]:
        raise ValueError(
            "expected the face list vertex_indices as the only list property, "
            f"got {lists}"
        )
    return texture, elements[0][1], elements[0][2], elements[1][1], elements[1][2]


def _read_color_ply(path: Path) -> ObjectMesh:
    """The mesh of a textured ASCII PLY, each face colored by its texture at the
    face's mean UV."""
    data = path.read_bytes()
    body = data.index(b"\n", data.index(b"end_header")) + 1
    texture_name, num_vertices, properties, num_faces, face_properties = _ply_layout(
        data[:body].decode("ascii").splitlines()
    )
    values = np.array(data[body:].split(), dtype=np.float64)
    vertices = values[: num_vertices * len(properties)].reshape(num_vertices, -1)
    face_values = values[num_vertices * len(properties) :]
    # A scalar property takes one value per face, and vertex_indices takes four
    # on a triangle: its length 3 and the 3 indices.
    row_width = len(face_properties) + 3
    length_column = face_properties.index("vertex_indices")
    if face_values.size != num_faces * row_width:
        raise ValueError(f"{path} is not a mesh of {num_faces} triangles")
    face_rows = face_values.reshape(num_faces, row_width)
    if (face_rows[:, length_column] != 3).any():
        raise ValueError(f"{path} is not a mesh of {num_faces} triangles")
    faces = face_rows[:, length_column + 1 : length_column + 4].astype(np.int64)
    texture_bgr = cv2.imread(str(path.parent / texture_name), cv2.IMREAD_COLOR)
    if texture_bgr is None:
        raise ValueError(f"could not read the texture {texture_name} of {path}")
    texture = cv2.cvtColor(texture_bgr, cv2.COLOR_BGR2RGB)
    height, width = texture.shape[:2]
    uv = vertices[:, [properties.index("texture_u"), properties.index("texture_v")]]
    u, v = uv[faces].mean(axis=1).T
    # texture_v runs up from the image's bottom row.
    columns = np.clip(np.round(u * (width - 1)), 0, width - 1).astype(np.int64)
    rows = np.clip(np.round((1.0 - v) * (height - 1)), 0, height - 1).astype(np.int64)
    xyz = [properties.index(axis) for axis in ("x", "y", "z")]
    return ObjectMesh(
        vertices_mm=vertices[:, xyz],
        faces=faces,
        face_colors=texture[rows, columns],
    )


def color_model_path(
    object_alias: str, cache_dir: str | Path | None = None
) -> Path | None:
    """Where :func:`fetch_hot3d_models` puts the color model of an object alias in
    ``cache_dir`` (default :func:`default_cache_dir`); None for an alias without a
    HOT3D mesh."""
    bop_id = HOT3D_BOP_OBJECT_ID.get(object_alias)
    if bop_id is None:
        return None
    folder = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    return folder / f"{_model_name(bop_id)}.ply"


def load_color_mesh(
    object_alias: str, cache_dir: str | Path | None = None
) -> ObjectMesh | None:
    """The fetched color model of an object alias, with ``face_colors``; None for
    an alias without a HOT3D mesh, or when :func:`fetch_hot3d_models` has not put
    the model in ``cache_dir`` (default :func:`default_cache_dir`).

    The largest model, birdhousetoy, has 386k faces and takes a few seconds to
    read, so load a model once and reuse it.
    """
    path = color_model_path(object_alias, cache_dir)
    return _read_color_ply(path) if path is not None and path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch the color models of the 22 HOT3D objects from BOP-HOT3D"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=f"where to put the models (default: {default_cache_dir()})",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=None,
        help="a local copy of hot3d_models.zip to extract instead of downloading",
    )
    args = parser.parse_args()
    fetch_hot3d_models(args.cache_dir, zip_path=args.zip, verbose=True)


if __name__ == "__main__":
    main()

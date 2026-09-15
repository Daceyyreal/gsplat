# SPDX-FileCopyrightText: Copyright 2024 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

import json
import math
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

device = torch.device("cuda:0")

PNG_PARAMS = ["means", "scales", "quats", "opacities", "sh0"]


def _smooth_splats(n_sidelen: int, n_outliers: int = 0, seed: int = 0):
    """Spatially coherent splats on an n_sidelen x n_sidelen grid, i.e. already in the
    order that sorting would produce, with optional outliers in every PNG-coded param."""
    gen = torch.Generator().manual_seed(seed)
    n = n_sidelen**2
    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, n_sidelen), torch.linspace(0, 1, n_sidelen), indexing="ij"
    )

    def field(channels: int) -> torch.Tensor:
        freqs = torch.rand(channels, 2, generator=gen) * 1.5 + 0.5
        phases = torch.rand(channels, generator=gen) * 2 * math.pi
        waves = [
            torch.sin(2 * math.pi * (f[0] * xs + f[1] * ys) + p)
            for f, p in zip(freqs, phases)
        ]
        return torch.stack(waves, dim=-1).reshape(n, channels)

    splats = {
        "means": field(3) * 3.0,
        "scales": field(3) - 3.0,
        "quats": field(4) + 0.2,
        "opacities": field(1)[:, 0] * 2.0,
        "sh0": field(3).reshape(n, 1, 3) * 0.5,
    }
    outlier_idx = torch.randperm(n, generator=gen)[:n_outliers]
    outlier_value = {
        "means": 1e4,
        "scales": 5.0,
        "quats": 50.0,
        "opacities": -30.0,
        "sh0": 20.0,
    }
    for k, v in outlier_value.items():
        splats[k].reshape(n, -1)[outlier_idx, 0] = v
    return splats, outlier_idx


def _reference_png_compress(compress_dir: str, splats):
    """Frozen copy of the PngCompression format before `tile_size` and `bits` were
    added (without sorting, cropping and K-means), used to check that the default
    settings still write the same bytes."""
    import imageio.v2 as imageio
    from gsplat.utils import log_transform

    splats = dict(splats)
    splats["means"] = log_transform(splats["means"])
    splats["quats"] = F.normalize(splats["quats"], dim=-1)
    n_sidelen = int(len(splats["means"]) ** 0.5)
    meta = {}
    for name, params in splats.items():
        grid = params.reshape((n_sidelen, n_sidelen, -1))
        mins = torch.amin(grid, dim=(0, 1))
        maxs = torch.amax(grid, dim=(0, 1))
        img_norm = ((grid - mins) / (maxs - mins)).detach().cpu().numpy()
        if name == "means":
            img = (img_norm * (2**16 - 1)).round().astype(np.uint16)
            imageio.imwrite(
                os.path.join(compress_dir, f"{name}_l.png"),
                (img & 0xFF).astype(np.uint8),
            )
            imageio.imwrite(
                os.path.join(compress_dir, f"{name}_u.png"),
                ((img >> 8) & 0xFF).astype(np.uint8),
            )
        else:
            img = (img_norm * (2**8 - 1)).round().astype(np.uint8).squeeze()
            imageio.imwrite(os.path.join(compress_dir, f"{name}.png"), img)
        meta[name] = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
            "mins": mins.tolist(),
            "maxs": maxs.tolist(),
        }
    with open(os.path.join(compress_dir, "meta.json"), "w") as f:
        json.dump(meta, f)


def _clone(splats):
    return {k: v.clone() for k, v in splats.items()}


def test_png_compression_default_is_unchanged(tmp_path):
    pytest.importorskip("imageio")
    from gsplat.compression import PngCompression

    splats, _ = _smooth_splats(64, n_outliers=4)
    ref_dir, new_dir = tmp_path / "reference", tmp_path / "new"
    ref_dir.mkdir()
    new_dir.mkdir()
    _reference_png_compress(str(ref_dir), _clone(splats))
    PngCompression(use_sort=False, verbose=False).compress(str(new_dir), _clone(splats))

    ref_files = sorted(os.listdir(ref_dir))
    assert ref_files == sorted(os.listdir(new_dir))
    for name in ref_files:
        assert (ref_dir / name).read_bytes() == (new_dir / name).read_bytes(), name


def test_png_compression_decodes_legacy_meta(tmp_path):
    pytest.importorskip("imageio")
    from gsplat.compression import PngCompression
    from gsplat.utils import log_transform

    splats, _ = _smooth_splats(32)
    _reference_png_compress(str(tmp_path), _clone(splats))
    with open(tmp_path / "meta.json") as f:
        assert all("bits" not in m for m in json.load(f).values())

    # Settings of the decoding instance must not matter for old directories.
    decoded = PngCompression(tile_size=8, bits={"means": 10, "sh0": 6}).decompress(
        str(tmp_path)
    )
    expected = PngCompression().decompress(str(tmp_path))
    for k in PNG_PARAMS:
        assert torch.equal(decoded[k], expected[k]), k

    # Old format error bounds: 16-bit means (in log space), 8-bit others.
    targets = _clone(splats)
    targets["quats"] = F.normalize(targets["quats"], dim=-1)
    for k in PNG_PARAMS:
        target, value = targets[k], decoded[k]
        if k == "means":
            target, value = log_transform(target), log_transform(value)
        bits = 16 if k == "means" else 8
        ranges = target.reshape(len(target), -1).amax(0) - target.reshape(
            len(target), -1
        ).amin(0)
        err = (value - target).reshape(len(target), -1).abs().amax(0)
        assert torch.all(err <= ranges / (2**bits - 1) / 2 + 1e-5), k


@pytest.mark.parametrize("tile_size", [8, 16])
def test_png_compression_tiles_reduce_outlier_error(tmp_path, tile_size):
    pytest.importorskip("imageio")
    from gsplat.compression import PngCompression

    splats, outlier_idx = _smooth_splats(128, n_outliers=6)
    inliers = torch.ones(len(splats["means"]), dtype=torch.bool)
    inliers[outlier_idx] = False
    targets = _clone(splats)
    targets["quats"] = F.normalize(targets["quats"], dim=-1)

    errors = {}
    for name, method in [
        ("global", PngCompression(use_sort=False, verbose=False)),
        ("tile", PngCompression(use_sort=False, verbose=False, tile_size=tile_size)),
    ]:
        compress_dir = tmp_path / name
        compress_dir.mkdir()
        method.compress(str(compress_dir), _clone(splats))
        decoded = method.decompress(str(compress_dir))
        errors[name] = {
            k: F.mse_loss(decoded[k][inliers], targets[k][inliers]).item()
            for k in PNG_PARAMS
        }
        if name == "tile":
            for k in PNG_PARAMS:
                assert (compress_dir / f"{k}_tiles.npz").is_file()

    for k in PNG_PARAMS:
        assert errors["tile"][k] < errors["global"][k], (k, errors)


@pytest.mark.parametrize("bits", [1, 6, 8, 10, 12, 16])
@pytest.mark.parametrize("tile_size", [None, 1, 7, 16, 64])
def test_png_quant_round_trip_error_bound(tmp_path, bits, tile_size):
    pytest.importorskip("imageio")
    from gsplat.compression.png_compression import (
        _compress_png_quant,
        _decompress_png_quant,
    )

    # 50 is not divisible by 7 or 16, and smaller than 64.
    n_sidelen, channels = 50, 3
    gen = torch.Generator().manual_seed(bits)
    params = torch.randn(n_sidelen**2, channels, generator=gen) * 10
    meta = _compress_png_quant(
        str(tmp_path), "p", params, n_sidelen, bits=bits, tile_size=tile_size
    )
    decoded = _decompress_png_quant(str(tmp_path), "p", json.loads(json.dumps(meta)))
    assert decoded.shape == params.shape and decoded.dtype == params.dtype

    grid = params.double().reshape(n_sidelen, n_sidelen, channels)
    if tile_size is None:
        mins = torch.tensor(meta["mins"], dtype=torch.float64)
        maxs = torch.tensor(meta["maxs"], dtype=torch.float64)
        assert torch.equal(mins, grid.amin((0, 1)))
    else:
        n_tiles = -(-n_sidelen // tile_size)
        with np.load(tmp_path / "p_tiles.npz") as f:
            tiles = {k: f[k] for k in f.files}
        assert tiles["mins"].dtype == np.float16
        assert tiles["mins"].shape == (n_tiles, n_tiles, channels)
        idx = torch.arange(n_sidelen) // tile_size
        mins = torch.from_numpy(tiles["mins"].astype(np.float64))[idx][:, idx]
        maxs = torch.from_numpy(tiles["maxs"].astype(np.float64))[idx][:, idx]
        # Stored bounds are rounded outwards, so they contain every value.
        assert torch.all(mins <= grid) and torch.all(grid <= maxs)
    step = (maxs - mins) / (2**bits - 1)
    err = (decoded.double().reshape(grid.shape) - grid).abs()
    # float32 output adds up to ~1e-6 relative error on values of magnitude ~10
    assert torch.all(err <= step / 2 + 1e-5)


@pytest.mark.parametrize("bits", [6, 12])
def test_png_quant_file_layout(tmp_path, bits):
    imageio = pytest.importorskip("imageio.v2")
    from gsplat.compression import PngCompression

    splats, _ = _smooth_splats(16)
    method = PngCompression(
        use_sort=False, verbose=False, tile_size=4, bits={k: bits for k in PNG_PARAMS}
    )
    method.compress(str(tmp_path), _clone(splats))
    with open(tmp_path / "meta.json") as f:
        meta = json.load(f)
    for k in PNG_PARAMS:
        assert meta[k]["bits"] == bits and meta[k]["tile_size"] == 4
        if bits <= 8:
            assert not (tmp_path / f"{k}_u.png").exists()
            assert imageio.imread(tmp_path / f"{k}.png").max() < 2**bits
        else:
            assert not (tmp_path / f"{k}.png").exists()
            assert imageio.imread(tmp_path / f"{k}_l.png").max() < 2 ** (bits - 8)


def test_png_tile_bounds_fall_back_to_float32():
    from gsplat.compression.png_compression import _tile_bounds

    grid = np.zeros((8, 8, 2))
    grid[0, 0, 0] = 1e6  # does not fit into float16
    grid[5, 5, 1] = 0.1  # not representable in float16
    mins, maxs = _tile_bounds(grid, 4)
    assert mins.dtype == np.float32 and maxs[0, 0, 0] >= 1e6
    mins, maxs = _tile_bounds(grid[..., 1:], 4)
    assert mins.dtype == np.float16 and maxs[1, 1, 0] >= 0.1


def test_png_compression_invalid_args():
    from gsplat.compression import PngCompression

    with pytest.raises(ValueError):
        PngCompression(tile_size=0)
    with pytest.raises(ValueError):
        PngCompression(bits={"means": 17})
    with pytest.raises(ValueError):
        PngCompression(bits={"shN": 8})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_png_compression():
    from gsplat.compression import PngCompression

    torch.manual_seed(42)

    # Prepare Gaussians
    N = 100000
    splats = torch.nn.ParameterDict(
        {
            "means": torch.randn(N, 3),
            "scales": torch.randn(N, 3),
            "quats": torch.randn(N, 4),
            "opacities": torch.randn(N),
            "sh0": torch.randn(N, 1, 3),
            "shN": torch.randn(N, 24, 3),
            "features": torch.randn(N, 128),
        }
    ).to(device)
    compress_dir = "/tmp/gsplat/compression"

    compression_method = PngCompression()
    # run compression and save the compressed files to compress_dir
    compression_method.compress(compress_dir, splats)
    # decompress the compressed files
    splats_c = compression_method.decompress(compress_dir)


if __name__ == "__main__":
    test_png_compression()

# SPDX-FileCopyrightText: Copyright 2024-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

import functools
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat.compression.kmeans import weighted_kmeans
from gsplat.compression.sort import sort_splats
from gsplat.utils import inverse_log_transform, log_transform

# Quantization bit depth of the PNG-coded parameters in the original format.
_DEFAULT_PNG_BITS = {
    "means": 16,
    "scales": 8,
    "quats": 8,
    "opacities": 8,
    "sh0": 8,
}


@dataclass
class PngCompression:
    """Uses quantization and sorting to compress splats into PNG files and uses
    K-means clustering to compress the spherical harmonic coefficents.

    .. warning::
        This class requires the `imageio <https://pypi.org/project/imageio/>`_ and
        `plas <https://github.com/fraunhoferhhi/PLAS.git>`_ packages to be installed.
        The default K-means backend also requires
        `torchpq <https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install>`_;
        ``kmeans_backend="builtin"`` needs neither TorchPQ nor CuPy.

    .. warning::
        This class might throw away a few lowest opacities splats if the number of
        splats is not a square number.

    .. note::
        The splats parameters are expected to be pre-activation values. It expects
        the following fields in the splats dictionary: "means", "scales", "quats",
        "opacities", "sh0", "shN". More fields can be added to the dictionary, but
        they will only be compressed using NPZ compression.

    References:
        - `Compact 3D Scene Representation via Self-Organizing Gaussian Grids <https://arxiv.org/abs/2312.13299>`_
        - `Making Gaussian Splats more smaller <https://aras-p.info/blog/2023/09/27/Making-Gaussian-Splats-more-smaller/>`_

    Args:
        use_sort (bool, optional): Whether to sort splats before compression. Defaults to True.
        verbose (bool, optional): Whether to print verbose information. Default to True.
        kmeans_backend (str, optional): K-means implementation for the spherical harmonic
            coefficients: "torchpq" (default, unchanged behavior) or "builtin"
            (:func:`gsplat.compression.kmeans.weighted_kmeans`, no extra dependencies).
        kmeans_weighting (str, optional): Per-splat weighting of the K-means update, only
            used by the "builtin" backend: None (default, unweighted), "opacity"
            (``sigmoid(opacity)``) or "opacity_area" (``sigmoid(opacity)`` times the
            exponential of the two largest log-scales). Weighting spends centroids on the
            splats that cover more of the rendered image.
        tile_size (int, optional): If set, the PNG-coded parameters ("means", "scales",
            "quats", "opacities", "sh0") are quantized with a separate min/max per channel
            for every `tile_size` x `tile_size` block of the sorted grid, instead of one
            min/max per channel for the whole grid. This keeps a few outliers from
            stretching the quantization range of the entire scene. The per-tile bounds
            are stored in "{param}_tiles.npz". Blocks on the last row / column are
            smaller when the grid side is not divisible by `tile_size`. Default to None
            (one min/max per channel).
        bits (Dict[str, int], optional): Quantization bit depth (1 to 16) of the PNG-coded
            parameters, e.g. `{"means": 12, "sh0": 7}`. Parameters that are not listed
            use 16 bits for "means" and 8 bits for the others. Bit depths above 8 are
            stored in two 8-bit PNGs holding the upper 8 bits and the remaining lower
            bits. Default to None.

    .. note::
        With the default `tile_size` and `bits`, the compressed files are identical to
        earlier versions. Decompression reads the settings from "meta.json", so any
        instance can decompress directories written with any settings.
    """

    use_sort: bool = True
    verbose: bool = True
    kmeans_backend: str = "torchpq"
    kmeans_weighting: Optional[str] = None
    tile_size: Optional[int] = None
    bits: Optional[Dict[str, int]] = None

    def __post_init__(self):
        if self.tile_size is not None and self.tile_size < 1:
            raise ValueError(f"tile_size must be positive, got {self.tile_size}")
        for param_name, bits in (self.bits or {}).items():
            if param_name not in _DEFAULT_PNG_BITS:
                raise ValueError(
                    f"bits can only be set for {list(_DEFAULT_PNG_BITS)}, got '{param_name}'"
                )
            if not 1 <= bits <= 16:
                raise ValueError(
                    f"bits for '{param_name}' must be in [1, 16], got {bits}"
                )

    def _get_compress_fn(self, param_name: str) -> Callable:
        if param_name in _DEFAULT_PNG_BITS:
            bits = (self.bits or {}).get(param_name, _DEFAULT_PNG_BITS[param_name])
            if self.tile_size is not None or bits != _DEFAULT_PNG_BITS[param_name]:
                return functools.partial(
                    _compress_png_quant, bits=bits, tile_size=self.tile_size
                )

        compress_fn_map = {
            "means": _compress_png_16bit,
            "scales": _compress_png,
            "quats": _compress_png,
            "opacities": _compress_png,
            "sh0": _compress_png,
            "shN": _compress_kmeans,
        }
        if param_name in compress_fn_map:
            return compress_fn_map[param_name]
        else:
            return _compress_npz

    def _get_decompress_fn(
        self, param_name: str, param_meta: Optional[Dict[str, Any]] = None
    ) -> Callable:
        # Only the configurable quantization format records "bits" in its metadata.
        if param_meta is not None and "bits" in param_meta:
            return _decompress_png_quant

        decompress_fn_map = {
            "means": _decompress_png_16bit,
            "scales": _decompress_png,
            "quats": _decompress_png,
            "opacities": _decompress_png,
            "sh0": _decompress_png,
            "shN": _decompress_kmeans,
        }
        if param_name in decompress_fn_map:
            return decompress_fn_map[param_name]
        else:
            return _decompress_npz

    def compress(self, compress_dir: str, splats: Dict[str, Tensor]) -> None:
        """Run compression

        Args:
            compress_dir (str): directory to save compressed files
            splats (Dict[str, Tensor]): Gaussian splats to compress
        """

        # Param-specific preprocessing
        splats["means"] = log_transform(splats["means"])
        splats["quats"] = F.normalize(splats["quats"], dim=-1)

        n_gs = len(splats["means"])
        n_sidelen = int(n_gs**0.5)
        n_crop = n_gs - n_sidelen**2
        if n_crop != 0:
            splats = _crop_n_splats(splats, n_crop)
            print(
                f"Warning: Number of Gaussians was not square. Removed {n_crop} Gaussians."
            )

        if self.use_sort:
            splats = sort_splats(splats)

        # After sorting, so that the weights line up with the splats that are clustered.
        kmeans_weights = _kmeans_weights(splats, self.kmeans_weighting)

        meta = {}
        for param_name in splats.keys():
            compress_fn = self._get_compress_fn(param_name)
            kwargs = {
                "n_sidelen": n_sidelen,
                "verbose": self.verbose,
                "weights": kmeans_weights,
                "backend": self.kmeans_backend,
            }
            meta[param_name] = compress_fn(
                compress_dir, param_name, splats[param_name], **kwargs
            )

        with open(os.path.join(compress_dir, "meta.json"), "w") as f:
            json.dump(meta, f)

    def decompress(self, compress_dir: str) -> Dict[str, Tensor]:
        """Run decompression

        Args:
            compress_dir (str): directory that contains compressed files

        Returns:
            Dict[str, Tensor]: decompressed Gaussian splats
        """
        with open(os.path.join(compress_dir, "meta.json"), "r") as f:
            meta = json.load(f)

        splats = {}
        for param_name, param_meta in meta.items():
            decompress_fn = self._get_decompress_fn(param_name, param_meta)
            splats[param_name] = decompress_fn(compress_dir, param_name, param_meta)

        # Param-specific postprocessing
        splats["means"] = inverse_log_transform(splats["means"])
        return splats


def _kmeans_weights(
    splats: Dict[str, Tensor], weighting: Optional[str]
) -> Optional[Tensor]:
    """Per-splat K-means weights, computed from pre-activation opacities and log-scales.

    Args:
        splats (Dict[str, Tensor]): splats, after sorting.
        weighting (str, optional): None, "opacity" or "opacity_area".

    Returns:
        Tensor, optional: weights [N], or None when unweighted.
    """
    if weighting is None:
        return None
    opacities = torch.sigmoid(splats["opacities"].reshape(-1).double())
    if weighting == "opacity":
        return opacities
    if weighting == "opacity_area":
        # exp of the two largest log-scales: the extent of the splat's largest cross section,
        # up to a constant factor.
        log_scales = splats["scales"].double()
        return opacities * torch.exp(log_scales.topk(2, dim=-1).values.sum(dim=-1))
    raise ValueError(f"Unknown kmeans_weighting: {weighting}")


def _crop_n_splats(splats: Dict[str, Tensor], n_crop: int) -> Dict[str, Tensor]:
    opacities = splats["opacities"]
    keep_indices = torch.argsort(opacities, descending=True)[:-n_crop]
    for k, v in splats.items():
        splats[k] = v[keep_indices]
    return splats


def _compress_png(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, **kwargs
) -> Dict[str, Any]:
    """Compress parameters with 8-bit quantization and lossless PNG compression.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        params (Tensor): parameters
        n_sidelen (int): image side length

    Returns:
        Dict[str, Any]: metadata
    """
    import imageio.v2 as imageio

    if torch.numel == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
        }
        return meta

    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)
    img_norm = grid_norm.detach().cpu().numpy()

    img = (img_norm * (2**8 - 1)).round().astype(np.uint8)
    img = img.squeeze()
    imageio.imwrite(os.path.join(compress_dir, f"{param_name}.png"), img)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    }
    return meta


def _decompress_png(compress_dir: str, param_name: str, meta: Dict[str, Any]) -> Tensor:
    """Decompress parameters from PNG file.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """
    import imageio.v2 as imageio

    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return meta

    img = imageio.imread(os.path.join(compress_dir, f"{param_name}.png"))
    img_norm = img / (2**8 - 1)

    grid_norm = torch.tensor(img_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_png_16bit(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, **kwargs
) -> Dict[str, Any]:
    """Compress parameters with 16-bit quantization and PNG compression.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        params (Tensor): parameters
        n_sidelen (int): image side length

    Returns:
        Dict[str, Any]: metadata
    """
    import imageio.v2 as imageio

    if torch.numel == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
        }
        return meta

    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)
    img_norm = grid_norm.detach().cpu().numpy()
    img = (img_norm * (2**16 - 1)).round().astype(np.uint16)

    img_l = img & 0xFF
    img_u = (img >> 8) & 0xFF
    imageio.imwrite(
        os.path.join(compress_dir, f"{param_name}_l.png"), img_l.astype(np.uint8)
    )
    imageio.imwrite(
        os.path.join(compress_dir, f"{param_name}_u.png"), img_u.astype(np.uint8)
    )

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    }
    return meta


def _decompress_png_16bit(
    compress_dir: str, param_name: str, meta: Dict[str, Any]
) -> Tensor:
    """Decompress parameters from PNG files.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """
    import imageio.v2 as imageio

    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return meta

    img_l = imageio.imread(os.path.join(compress_dir, f"{param_name}_l.png"))
    img_u = imageio.imread(os.path.join(compress_dir, f"{param_name}_u.png"))
    img_u = img_u.astype(np.uint16)
    img = (img_u << 8) + img_l

    img_norm = img / (2**16 - 1)
    grid_norm = torch.tensor(img_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_png_quant(
    compress_dir: str,
    param_name: str,
    params: Tensor,
    n_sidelen: int,
    bits: int,
    tile_size: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Compress parameters with `bits`-bit quantization and lossless PNG compression.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        params (Tensor): parameters
        n_sidelen (int): image side length
        bits (int): number of quantization bits, from 1 to 16
        tile_size (int, optional): side length of the blocks that get their own
            per-channel min/max. If None, one min/max per channel is used.

    Returns:
        Dict[str, Any]: metadata
    """
    import imageio.v2 as imageio

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "bits": bits,
    }
    if params.numel() == 0:
        if tile_size is not None:
            meta["tile_size"] = tile_size
        return meta

    grid = params.detach().reshape((n_sidelen, n_sidelen, -1)).cpu().double().numpy()
    if tile_size is None:
        mins = grid.min(axis=(0, 1))
        maxs = grid.max(axis=(0, 1))
        meta["mins"] = mins.tolist()
        meta["maxs"] = maxs.tolist()
    else:
        tile_mins, tile_maxs = _tile_bounds(grid, tile_size)
        np.savez_compressed(
            os.path.join(compress_dir, f"{param_name}_tiles.npz"),
            mins=tile_mins,
            maxs=tile_maxs,
        )
        meta["tile_size"] = tile_size
        mins = _expand_tiles(tile_mins, tile_size, n_sidelen)
        maxs = _expand_tiles(tile_maxs, tile_size, n_sidelen)

    ranges = maxs - mins
    grid_norm = (grid - mins) / np.where(ranges > 0, ranges, 1.0)
    img = (np.clip(grid_norm, 0.0, 1.0) * (2**bits - 1)).round().astype(np.uint16)
    if img.shape[-1] == 1:
        img = img[..., 0]

    if bits <= 8:
        imageio.imwrite(
            os.path.join(compress_dir, f"{param_name}.png"), img.astype(np.uint8)
        )
    else:
        n_lower = bits - 8
        img_u = img >> n_lower
        img_l = img & ((1 << n_lower) - 1)
        imageio.imwrite(
            os.path.join(compress_dir, f"{param_name}_l.png"), img_l.astype(np.uint8)
        )
        imageio.imwrite(
            os.path.join(compress_dir, f"{param_name}_u.png"), img_u.astype(np.uint8)
        )
    return meta


def _decompress_png_quant(
    compress_dir: str, param_name: str, meta: Dict[str, Any]
) -> Tensor:
    """Decompress parameters written by :func:`_compress_png_quant`.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """
    import imageio.v2 as imageio

    if not np.all(meta["shape"]):
        return torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))

    bits = meta["bits"]
    if bits <= 8:
        img = imageio.imread(os.path.join(compress_dir, f"{param_name}.png"))
    else:
        img_l = imageio.imread(os.path.join(compress_dir, f"{param_name}_l.png"))
        img_u = imageio.imread(os.path.join(compress_dir, f"{param_name}_u.png"))
        img = (img_u.astype(np.uint16) << (bits - 8)) | img_l.astype(np.uint16)
    img = img.reshape((img.shape[0], img.shape[1], -1))
    n_sidelen = img.shape[0]

    if "tile_size" in meta:
        with np.load(os.path.join(compress_dir, f"{param_name}_tiles.npz")) as tiles:
            mins = _expand_tiles(tiles["mins"], meta["tile_size"], n_sidelen)
            maxs = _expand_tiles(tiles["maxs"], meta["tile_size"], n_sidelen)
    else:
        mins = np.array(meta["mins"], dtype=np.float64)
        maxs = np.array(meta["maxs"], dtype=np.float64)

    grid = img / (2**bits - 1) * (maxs - mins) + mins
    params = torch.from_numpy(grid).reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _tile_bounds(grid: np.ndarray, tile_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-tile, per-channel min and max of a [H, W, C] grid.

    The bounds are rounded outwards to float16, so that every value stays inside the
    stored range. Float32 is used instead if a bound does not fit into float16.

    Returns:
        Tuple[np.ndarray, np.ndarray]: mins and maxs, each [ceil(H / tile_size),
        ceil(W / tile_size), C]
    """
    starts_h = np.arange(0, grid.shape[0], tile_size)
    starts_w = np.arange(0, grid.shape[1], tile_size)
    mins = np.minimum.reduceat(grid, starts_h, axis=0)
    mins = np.minimum.reduceat(mins, starts_w, axis=1)
    maxs = np.maximum.reduceat(grid, starts_h, axis=0)
    maxs = np.maximum.reduceat(maxs, starts_w, axis=1)

    for dtype in (np.float16, np.float32):
        with np.errstate(over="ignore"):
            mins_c = mins.astype(dtype)
            maxs_c = maxs.astype(dtype)
        mins_c = np.where(mins_c > mins, np.nextafter(mins_c, dtype(-np.inf)), mins_c)
        maxs_c = np.where(maxs_c < maxs, np.nextafter(maxs_c, dtype(np.inf)), maxs_c)
        if np.isfinite(mins_c).all() and np.isfinite(maxs_c).all():
            break
    return mins_c, maxs_c


def _expand_tiles(tiles: np.ndarray, tile_size: int, n_sidelen: int) -> np.ndarray:
    """Expand [T, T, C] per-tile values to a [n_sidelen, n_sidelen, C] float64 grid."""
    idx = np.arange(n_sidelen) // tile_size
    return tiles.astype(np.float64)[idx[:, None], idx[None, :]]


def _compress_npz(
    compress_dir: str, param_name: str, params: Tensor, **kwargs
) -> Dict[str, Any]:
    """Compress parameters with numpy's NPZ compression."""
    npz_dict = {"arr": params.detach().cpu().numpy()}
    save_fp = os.path.join(compress_dir, f"{param_name}.npz")
    os.makedirs(os.path.dirname(save_fp), exist_ok=True)
    np.savez_compressed(save_fp, **npz_dict)
    meta = {
        "shape": params.shape,
        "dtype": str(params.dtype).split(".")[1],
    }
    return meta


def _decompress_npz(compress_dir: str, param_name: str, meta: Dict[str, Any]) -> Tensor:
    """Decompress parameters with numpy's NPZ compression."""
    arr = np.load(os.path.join(compress_dir, f"{param_name}.npz"))["arr"]
    params = torch.tensor(arr)
    params = params.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_kmeans(
    compress_dir: str,
    param_name: str,
    params: Tensor,
    n_clusters: int = 65536,
    quantization: int = 6,
    eps: float = 1e-6,
    verbose: bool = True,
    weights: Optional[Tensor] = None,
    backend: str = "torchpq",
    **kwargs,
) -> Dict[str, Any]:
    """Run K-means clustering on parameters and save centroids and labels to a npz file.

    .. warning::
        The "torchpq" backend requires TorchPQ to be installed.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        params (Tensor): parameters to compress
        n_clusters (int): number of K-means clusters
        quantization (int): number of bits in quantization
        eps (float, optional): small value to avoid numerical issues. Default to 1e-6.
        verbose (bool, optional): Whether to print verbose information. Default to True.
        weights (Tensor, optional): per-splat weights [N] for the centroid update. Only used
            by the "builtin" backend. Default to None (unweighted).
        backend (str, optional): "torchpq" (default) or "builtin"
            (:func:`gsplat.compression.kmeans.weighted_kmeans`). Default to "torchpq".

    Returns:
        Dict[str, Any]: metadata
    """
    if torch.numel == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
        }
        return meta

    if backend == "torchpq":
        if weights is not None:
            raise ValueError(
                "K-means weights are only supported by the 'builtin' backend"
            )
        try:
            from torchpq.clustering import KMeans
        except:
            raise ImportError(
                "Please install extra dependencies with 'pip install torchpq cupy' to use K-means clustering"
            )

        kmeans = KMeans(n_clusters=n_clusters, distance="manhattan", verbose=verbose)
        x = params.reshape(params.shape[0], -1).permute(1, 0).contiguous()
        labels = kmeans.fit(x)
        labels = labels.detach().cpu().numpy()
        centroids = kmeans.centroids.permute(1, 0)
    elif backend == "builtin":
        x = params.reshape(params.shape[0], -1)
        centroids, labels_t = weighted_kmeans(
            x, min(n_clusters, x.shape[0]), weights=weights
        )
        labels = labels_t.detach().cpu().numpy()
        # Same memory layout as the TorchPQ path (a [K, D] view of a [D, K] buffer), so that
        # the npz bytes depend on the values only, not on the backend.
        centroids = centroids.t().contiguous().t()
    else:
        raise ValueError(f"Unknown K-means backend: {backend}")

    mins = torch.min(centroids) + eps
    maxs = torch.max(centroids)
    centroids_norm = (centroids - mins) / (maxs - mins)
    centroids_norm = centroids_norm.detach().cpu().numpy()
    centroids_quant = np.asfortranarray(
        (centroids_norm * (2**quantization - 1)).round().astype(np.uint8)
    )
    labels = labels.astype(np.uint16)

    npz_dict = {
        "centroids": centroids_quant,
        "labels": labels,
    }
    np.savez_compressed(os.path.join(compress_dir, f"{param_name}.npz"), **npz_dict)
    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "quantization": quantization,
    }
    return meta


def _decompress_kmeans(
    compress_dir: str, param_name: str, meta: Dict[str, Any], **kwargs
) -> Tensor:
    """Decompress parameters from K-means compression.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """
    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return meta

    npz_dict = np.load(os.path.join(compress_dir, f"{param_name}.npz"))
    centroids_quant = npz_dict["centroids"]
    labels = npz_dict["labels"]

    centroids_norm = centroids_quant / (2 ** meta["quantization"] - 1)
    centroids_norm = torch.tensor(centroids_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    centroids = centroids_norm * (maxs - mins) + mins

    params = centroids[labels]
    params = params.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params

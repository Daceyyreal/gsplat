"""Gauss-Newton metric for the shN coefficients (definitions in kaggle/PREREG_GN.md).

Per splat i, over the train views v where it is visible:

    M_i = sum_v s_iv * y(d_iv) y(d_iv)^T      (15 x 15, bands 1-3; stored as the upper triangle)
    s_iv = sum_p w_ip^2                       (16-probe Hutchinson estimate)
    F_i = sum_v f_iv,  f_iv = sum_p w_ip      (exact: the all-ones channel)

``w_ip`` is splat i's blending weight at pixel p, ``y(d)`` the SH basis of bands 1-3 at
``d_iv = normalize(mean_i - campos_v)``. Both come from one render of zero features ``[N, 17]`` with
``sh_degree=None`` and the eval's rasterize settings: backpropagating Rademacher +-1 on channels 0-15
and 1 on channel 16 gives ``sum_p w_ip r_p`` per channel.
"""

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sh_basis as sb  # noqa: E402

N_PROBES = 16  # Rademacher channels per render
N_CHANNELS = (
    N_PROBES + 1
)  # + the all-ones channel; 17 is in gsplat's compiled channel list
D = sb.N_SHN  # 15
TRIU_I, TRIU_J = torch.triu_indices(D, D)  # 120 upper-triangle entries, row-major
N_PACKED = len(TRIU_I)
DIAG = (TRIU_I == TRIU_J).nonzero(as_tuple=True)[
    0
]  # positions of the diagonal in the packing
CACHE_VERSION = 1


# ------------------------------------------------------------------ packed 15 x 15 matrices


def pack_outer(y: Tensor) -> Tensor:
    """Upper triangle of ``y y^T`` per row: ``[n, 15] -> [n, 120]``."""
    i, j = TRIU_I.to(y.device), TRIU_J.to(y.device)
    return y[:, i] * y[:, j]


def unpack(packed: Tensor) -> Tensor:
    """``[n, 120] -> [n, 15, 15]`` symmetric."""
    i, j = TRIU_I.to(packed.device), TRIU_J.to(packed.device)
    full = packed.new_zeros(packed.shape[0], D, D)
    full[:, i, j] = packed
    full[:, j, i] = packed
    return full


def pack(full: Tensor) -> Tensor:
    """``[n, 15, 15] -> [n, 120]`` (upper triangle)."""
    i, j = TRIU_I.to(full.device), TRIU_J.to(full.device)
    return full[:, i, j]


def trace_packed(packed: Tensor) -> Tensor:
    return packed[:, DIAG.to(packed.device)].sum(dim=-1)


def frobenius_weights(device, dtype=torch.float32) -> Tensor:
    """``<A, B>_F = sum(w * pack(A) * pack(B))`` for symmetric A, B: 1 on the diagonal, 2 off it."""
    return torch.where(TRIU_I == TRIU_J, 1.0, 2.0).to(device=device, dtype=dtype)


# ------------------------------------------------------------------------- rendering


@dataclass(frozen=True)
class RenderSettings:
    """The rasterize settings of ``simple_trainer.Runner.rasterize_splats`` for eval."""

    packed: bool
    rasterize_mode: str
    camera_model: str
    with_ut: bool
    with_eval3d: bool
    near_plane: float
    far_plane: float
    sh_degree: int

    @classmethod
    def from_cfg(cls, cfg) -> "RenderSettings":
        if getattr(cfg, "app_opt", False):
            raise ValueError(
                "app_opt changes the color model; the GN metric assumes SH colors"
            )
        if getattr(cfg, "post_processing", None) is not None:
            raise ValueError("post_processing is not part of the GN model")
        if getattr(cfg, "sh_fp16", False):
            raise ValueError("sh_fp16 is not modeled")
        return cls(
            packed=bool(cfg.packed),
            rasterize_mode="antialiased" if cfg.antialiased else "classic",
            camera_model=str(cfg.camera_model),
            with_ut=bool(cfg.with_ut),
            with_eval3d=bool(cfg.with_eval3d),
            near_plane=float(cfg.near_plane),
            far_plane=float(cfg.far_plane),
            sh_degree=int(cfg.sh_degree),
        )

    def as_dict(self) -> Dict:
        return asdict(self)


def activated(splats: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """What ``rasterize_splats`` passes to ``rasterization`` (detached)."""
    return {
        "means": splats["means"].detach(),
        "quats": splats["quats"].detach(),
        "scales": torch.exp(splats["scales"].detach()),
        "opacities": torch.sigmoid(splats["opacities"].detach()),
    }


def viewmat_of(camtoworld: Tensor) -> Tensor:
    """World-to-camera ``[1, 4, 4]``, computed as ``rasterize_splats`` does."""
    return torch.linalg.inv_ex(camtoworld.reshape(1, 4, 4)).inverse


def gsplat_render(
    act: Dict[str, Tensor],
    colors: Tensor,
    camtoworld: Tensor,
    K: Tensor,
    width: int,
    height: int,
    settings: RenderSettings,
    sh_degree: Optional[int] = None,
) -> Tuple[Tensor, Dict]:
    """``gsplat.rasterization`` with the eval's settings; ``[H, W, D]`` and the info dict.

    Background 0 (the eval passes none). ``sparse_grad`` is left False: it only changes the layout of
    parameter gradients, and the GN pass needs dense feature gradients."""
    from gsplat import rasterization

    img, _, info = rasterization(
        means=act["means"],
        quats=act["quats"],
        scales=act["scales"],
        opacities=act["opacities"],
        colors=colors,
        viewmats=viewmat_of(camtoworld),
        Ks=K.reshape(1, 3, 3),
        width=width,
        height=height,
        near_plane=settings.near_plane,
        far_plane=settings.far_plane,
        sh_degree=sh_degree,
        packed=settings.packed,
        sparse_grad=False,
        absgrad=False,
        rasterize_mode=settings.rasterize_mode,
        camera_model=settings.camera_model,
        with_ut=settings.with_ut,
        with_eval3d=settings.with_eval3d,
    )
    return img[0], info


def visible_from_info(info: Dict, n: int) -> Tensor:
    """Boolean ``[n]``: the splat has a radius > 0 in this (single) view."""
    gids = info.get("gaussian_ids")
    if gids is not None:  # packed
        vis = torch.zeros(n, dtype=torch.bool, device=gids.device)
        vis[gids.long()] = True
        return vis
    radii = info["radii"].reshape(n, -1)  # [1, N, 2] (or [1, N]) for one camera
    return (radii > 0).all(dim=-1)


def rademacher(shape, generator: torch.Generator, device) -> Tensor:
    return (
        torch.randint(0, 2, shape, generator=generator, device=device).float() * 2 - 1
    )


def probe_view(
    render_fn: Callable[[Tensor], Tuple[Tensor, Dict]],
    n: int,
    device,
    generator: torch.Generator,
    return_probes: bool = False,
):
    """One 17-channel render of zero features and one backward pass.

    Returns ``s [n]`` (mean of the 16 squared probe gradients), ``f [n]`` (all-ones channel),
    ``visible [n]``, the number of pixels and, with ``return_probes``, the ``[H, W, 17]`` pixel
    gradient that was backpropagated."""
    feats = torch.zeros(n, N_CHANNELS, device=device, requires_grad=True)
    img, info = render_fn(feats)
    h, w, c = img.shape
    if c != N_CHANNELS:
        raise RuntimeError(f"expected {N_CHANNELS} rendered channels, got {c}")
    grad = torch.empty_like(img)
    grad[..., :N_PROBES] = rademacher((h, w, N_PROBES), generator, device)
    grad[..., N_PROBES] = 1.0
    img.backward(grad)
    g = feats.grad
    s = g[:, :N_PROBES].pow(2).mean(dim=1)
    f = g[:, N_PROBES].clone()
    out = (s, f, visible_from_info(info, n), h * w)
    return out + (grad.detach(),) if return_probes else out


class GNAccumulator:
    """Per-splat sums over views: M (packed, fp32), F, the C3DGS-style |f y_k| sums, visibility."""

    def __init__(self, n: int, device, chunk: int = 262144):
        self.n, self.device, self.chunk = n, device, chunk
        self.M = torch.zeros(n, N_PACKED, dtype=torch.float32, device=device)
        self.F = torch.zeros(n, dtype=torch.float64, device=device)
        self.c3_abs = torch.zeros(n, D, dtype=torch.float32, device=device)
        self.s_sum = torch.zeros(n, dtype=torch.float64, device=device)
        self.n_vis = torch.zeros(n, dtype=torch.int32, device=device)
        self.clamp_neg = 0
        self.clamp_total = 0
        self.n_views = 0
        self.total_pixels = 0

    def add_view(
        self,
        s: Tensor,
        f: Tensor,
        visible: Tensor,
        means: Tensor,
        campos: Tensor,
        coeffs: Tensor,
        n_pixels: int,
    ) -> None:
        """``coeffs`` = ``cat(sh0, shN)`` ``[n, 16, 3]`` for the clamp statistic."""
        idx = visible.nonzero(as_tuple=True)[0]
        ones = torch.ones(
            min(self.chunk, max(len(idx), 1)), dtype=torch.int32, device=self.device
        )
        for part in idx.split(self.chunk):
            basis = sb.sh_basis(means[part] - campos, 3)  # [c, 16], normalized inside
            y = basis[:, 1:]
            self.M.index_add_(0, part, s[part, None] * pack_outer(y))
            self.F.index_add_(0, part, f[part].double())
            self.c3_abs.index_add_(0, part, (f[part, None] * y).abs())
            self.s_sum.index_add_(0, part, s[part].double())
            self.n_vis.index_add_(0, part, ones[: len(part)])
            color = (basis[:, :, None] * coeffs[part]).sum(dim=1) + 0.5  # [c, 3]
            self.clamp_neg += int((color < 0).sum())
            self.clamp_total += color.numel()
        self.n_views += 1
        self.total_pixels += int(n_pixels)

    def result(self) -> Dict:
        views = max(self.n_views, 1)
        return {
            "M_packed": self.M,
            "F": self.F,
            "c3dgs": (self.c3_abs / views).amax(dim=-1),
            "s_sum": self.s_sum,
            "n_vis": self.n_vis,
            "n_views": self.n_views,
            "total_pixels": self.total_pixels,
            "clamp_neg": self.clamp_neg,
            "clamp_total": self.clamp_total,
            "clamp_fraction": self.clamp_neg / max(self.clamp_total, 1),
        }


def compute_gn(
    splats: Dict[str, Tensor],
    views: Iterable[Dict],
    settings: RenderSettings,
    seed: int = 0,
    render: Optional[Callable] = None,
    chunk: int = 262144,
    log: Optional[Callable[[str], None]] = print,
    log_every: int = 20,
) -> Dict:
    """Accumulate the GN metric over ``views`` (dicts with ``camtoworld [4, 4]``, ``K [3, 3]``,
    ``width``, ``height``). ``render(act, colors, camtoworld, K, width, height, settings)`` must
    return ``([H, W, C], info)``; None means ``gsplat_render`` (looked up at call time)."""
    render = render or gsplat_render
    act = activated(splats)
    n = act["means"].shape[0]
    device = act["means"].device
    coeffs = torch.cat([splats["sh0"], splats["shN"]], dim=1).detach()
    gen = torch.Generator(device=device).manual_seed(seed)
    acc = GNAccumulator(n, device, chunk)
    t0 = time.perf_counter()
    for k, view in enumerate(views):
        c2w = view["camtoworld"].to(device)
        K = view["K"].to(device)
        W, H = int(view["width"]), int(view["height"])

        def render_fn(feats, c2w=c2w, K=K, W=W, H=H):
            return render(act, feats, c2w, K, W, H, settings)

        s, f, vis, n_pix = probe_view(render_fn, n, device, gen)
        campos = sb.camera_positions(viewmat_of(c2w))[0]
        acc.add_view(s, f, vis, act["means"], campos, coeffs, n_pix)
        if log is not None and (k + 1) % log_every == 0:
            log(f"GN: {k + 1} views, {time.perf_counter() - t0:.1f} s")
    out = acc.result()
    out["time_s"] = time.perf_counter() - t0
    out["probe_seed"] = seed
    out["settings"] = settings.as_dict()
    return out


def cache_key(ckpt_sha1: str, settings: RenderSettings, n_views: int, seed: int) -> str:
    return f"v{CACHE_VERSION}|{ckpt_sha1}|{sorted(settings.as_dict().items())}|views{n_views}|seed{seed}"


def save_cache(path: str, result: Dict, key: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in result.items()
    }
    payload["key"] = key
    torch.save(payload, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_cache(path: str, key: str, device) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("key") != key:
        return None
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in payload.items()}


# ------------------------------------------------------ scene fixtures (PREREG_GN.md Amendment 4)

# The toy and end-to-end checks render committed tensors, not fresh draws: torch's CPU randn differs
# in its last bits between dispatched CPU capabilities and platforms, so a fresh draw on another
# machine is not bitwise the pre-registered scene. Each fixture is an .npz (float32, little-endian)
# with a .json of metadata (hash, CPU capability, torch version, platform); bench/gn/make_fixtures.py
# wrote them.
FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
TOY_SCENE_FIXTURE = "toy_scene_seed0"
TOY_SCENE_SHA256 = "1bb442ee09b6d8417384f5aad10e019a3ebab15458141ef685c873d45f06e441"
FIXTURE_TOL_ABS = 1e-5  # a re-draw under another CPU capability or torch version


class FixtureHashMismatch(RuntimeError):
    """A scene fixture does not hash to its pinned value: a check would not render the
    pre-registered scene."""


def tensors_sha256(tensors: Dict[str, Tensor]) -> str:
    """SHA-256 over the tensors in sorted key order; for each, the bytes of
    ``"<key>:<shape>:float32-le;"`` (shape as a Python tuple), then its values as float32
    little-endian."""
    h = hashlib.sha256()
    for key in sorted(tensors):
        t = tensors[key].detach().to("cpu", torch.float32).contiguous()
        h.update(f"{key}:{tuple(t.shape)}:float32-le;".encode())
        h.update(t.numpy().astype("<f4", copy=False).tobytes())
    return h.hexdigest()


def cpu_environment() -> Dict[str, str]:
    """What a CPU draw's last bits depend on."""
    return {
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "torch_version": torch.__version__,
        "platform": sys.platform,
    }


def fixture_paths(name: str) -> Tuple[str, str]:
    base = os.path.join(FIXTURE_DIR, name)
    return base + ".npz", base + ".json"


def save_fixture(name: str, tensors: Dict[str, Tensor], meta: Dict) -> Dict:
    """Write ``<name>.npz`` and ``<name>.json``; returns the metadata."""
    import numpy as np

    npz, js = fixture_paths(name)
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    arrays = {
        k: v.detach().to("cpu", torch.float32).contiguous().numpy().astype("<f4")
        for k, v in tensors.items()
    }
    np.savez(npz, **arrays)
    full = {
        "name": name,
        "sha256": tensors_sha256(tensors),
        "hash": "SHA-256 over sorted keys: '<key>:<shape>:float32-le;' then float32 little-endian values",
        "shapes": {k: list(a.shape) for k, a in sorted(arrays.items())},
        **cpu_environment(),
        **meta,
    }
    with open(js, "w", newline="\n") as f:
        json.dump(full, f, indent=2)
        f.write("\n")
    return full


def load_fixture(
    name: str, expected_sha256: str, device="cpu"
) -> Tuple[Dict[str, Tensor], Dict]:
    """The fixture's tensors, moved to ``device`` only after their hash (and the one its metadata
    records) equals ``expected_sha256``; otherwise ``FixtureHashMismatch``."""
    import numpy as np

    npz, js = fixture_paths(name)
    with np.load(npz, allow_pickle=False) as z:
        tensors = {k: torch.from_numpy(z[k].astype(np.float32)) for k in z.files}
    with open(js) as f:
        meta = json.load(f)
    got = tensors_sha256(tensors)
    if got != expected_sha256 or meta.get("sha256") != expected_sha256:
        raise FixtureHashMismatch(
            f"scene fixture {name}: tensors hash to {got}, its metadata records "
            f"{meta.get('sha256')}, the pinned hash is {expected_sha256} (PREREG_GN.md Amendment 4)"
        )
    return {k: v.to(device) for k, v in tensors.items()}, meta


def fixture_provenance(
    fixture: Dict[str, Tensor],
    fresh: Dict[str, Tensor],
    meta: Dict,
    tol_abs: float = FIXTURE_TOL_ABS,
) -> Dict:
    """A fresh CPU draw against a fixture: bitwise equal when the fixture's CPU capability and torch
    version both match this machine, otherwise within ``tol_abs``."""
    here = cpu_environment()
    same = (
        meta.get("cpu_capability") == here["cpu_capability"]
        and meta.get("torch_version") == here["torch_version"]
    )
    out = {
        "mode": "bitwise" if same else "tolerance",
        "fixture_cpu_capability": meta.get("cpu_capability"),
        "fixture_torch_version": meta.get("torch_version"),
        "cpu_capability": here["cpu_capability"],
        "torch_version": here["torch_version"],
        "same_keys": sorted(fixture) == sorted(fresh),
    }
    if not out["same_keys"]:
        return {**out, "max_abs_diff": math.inf, "ok": False}
    diff = max(
        float((fixture[k].detach().cpu().double() - fresh[k].detach().cpu().double()).abs().max())
        for k in fixture
    )
    ok = tensors_sha256(fresh) == meta.get("sha256") if same else diff <= tol_abs
    return {**out, "max_abs_diff": diff, "tol_abs": tol_abs, "ok": bool(ok)}


# ------------------------------------------------------------------ toy exactness check


TOY_WIDTH, TOY_HEIGHT, TOY_FOCAL, TOY_DEPTH = 128, 96, 140.0, 3.0


def toy_scene(n: int = 256, seed: int = 0, device="cpu") -> Dict[str, Tensor]:
    """Pre-activation splats of similar size and opacity, spread uniformly over the toy view at
    depth 3 +- 0.3; most covered pixels blend two or more splats. Similar splats keep the
    64-probe noise of the summed estimate near sqrt(2 / 64) / sqrt(n) (see ``toy_exactness``).
    Drawn with the CPU generator, then moved to ``device`` (Amendment 4). The toy check renders the
    committed seed-0 draw (``TOY_SCENE_FIXTURE``), not a fresh one."""
    g = torch.Generator().manual_seed(seed)
    half_w = 0.9 * TOY_DEPTH * (TOY_WIDTH / 2) / TOY_FOCAL
    half_h = 0.9 * TOY_DEPTH * (TOY_HEIGHT / 2) / TOY_FOCAL
    xy = torch.rand(n, 2, generator=g) * 2 - 1
    z = TOY_DEPTH + torch.randn(n, generator=g) * 0.3
    splats = {
        "means": torch.stack([xy[:, 0] * half_w, xy[:, 1] * half_h, z], dim=-1),
        "quats": torch.randn(n, 4, generator=g),
        "scales": torch.randn(n, 3, generator=g) * 0.1 - 3.0,
        "opacities": torch.randn(n, generator=g) * 0.3 + 0.5,
        "sh0": torch.randn(n, 1, 3, generator=g) * 0.5,
        "shN": torch.randn(n, 15, 3, generator=g) * 0.2,
    }
    return {k: t.to(device) for k, t in splats.items()}


def toy_camera(device="cpu") -> Dict:
    K = torch.tensor(
        [
            [TOY_FOCAL, 0.0, TOY_WIDTH / 2],
            [0.0, TOY_FOCAL, TOY_HEIGHT / 2],
            [0.0, 0.0, 1.0],
        ],
        device=device,
    )
    return {
        "camtoworld": torch.eye(4, device=device),
        "K": K,
        "width": TOY_WIDTH,
        "height": TOY_HEIGHT,
    }


def hutchinson_rel_std(weights: Iterable[Tensor], n_probes: int) -> float:
    """Exact relative standard deviation of the ``n_probes``-probe Rademacher estimate of
    ``S = sum w^2`` (PREREG_GN.md Amendment 4), from exact weights per view given as
    ``[pixels, splats]``: ``sqrt(sum_views 2 (||W W^T||_F^2 - sum_p (sum_i w_ip^2)^2) / n_probes) / S``
    with ``W`` = splats x pixels (the variance of the quadratic form ``r^T W^T W r``)."""
    var, total = 0.0, 0.0
    for w in weights:
        w = w.double()
        gram = w.t() @ w  # W W^T, splats x splats
        per_pixel = w.pow(2).sum(dim=1)  # sum_i w_ip^2
        var += (
            2.0 * (float(gram.pow(2).sum()) - float(per_pixel.pow(2).sum())) / n_probes
        )
        total += float(per_pixel.sum())
    return math.sqrt(max(var, 0.0)) / total if total > 0 else math.nan


def false_fail_probability(sigma_rel: float, tol_rel: float = 0.05) -> float:
    """``P(|S_hat - S| >= tol_rel * S)`` for ``S_hat ~ Normal(S, (sigma_rel * S)^2)``."""
    if sigma_rel == 0:
        return 0.0
    return math.erfc(tol_rel / (math.sqrt(2.0) * sigma_rel))


def toy_exactness(
    render: Optional[Callable] = None,
    settings: Optional[RenderSettings] = None,
    n_renders: int = 4,
    seed: int = 0,
    device="cuda",
    tol_rel: float = 0.05,
    tol_exact: float = 1e-4,
    log: Optional[Callable[[str], None]] = print,
) -> Dict:
    """Hutchinson (``n_renders`` x 16 probes) against the exact ``s_i = sum_p w_ip^2`` from an
    identity-feature render (channel j of the render is ``w_pj``) on the toy scene.

    The scene is the committed fixture ``TOY_SCENE_FIXTURE`` (Amendment 4): loaded, its hash asserted
    before anything is rendered (``FixtureHashMismatch`` otherwise), then moved to ``device``.
    ``seed`` seeds the probes only.

    Pass needs all three:
    - the sum over splats of the 64-probe estimate within ``tol_rel`` of the exact sum (the
      pre-registered check; per-splat errors are reported);
    - the gradient-based estimate equal, to ``tol_exact`` relative, to the Hutchinson formula
      ``mean_c (sum_p w_pi r_pc)^2`` evaluated on the exact weights with the same probes: a
      deterministic test of the gradient plumbing;
    - the all-ones channel equal to ``sum_p w_pi`` to ``tol_exact`` relative.
    Relative errors per splat use the splats with ``sum_p w_pi > 1e-3 * max``.

    Report-only, outside every verdict and logged before the check is judged (Amendment 4):
    ``noise_diagnostic``, the exact relative std of the probe estimate of the sum
    (``hutchinson_rel_std``) and the implied false-fail probability of the ``tol_rel`` rule under a
    normal approximation."""
    render = render or gsplat_render
    settings = settings or RenderSettings(
        False, "classic", "pinhole", False, False, 0.01, 1e10, 3
    )
    splats, _ = load_fixture(TOY_SCENE_FIXTURE, TOY_SCENE_SHA256, device)
    n = splats["means"].shape[0]
    cam = toy_camera(device=device)
    act = activated(splats)

    def render_fn(feats):
        return render(
            act,
            feats,
            cam["camtoworld"],
            cam["K"],
            cam["width"],
            cam["height"],
            settings,
        )

    with torch.no_grad():
        ident, _ = render_fn(torch.eye(n, device=device))
    wts = ident.double().reshape(-1, n)  # [P, n]
    exact_s = wts.pow(2).sum(dim=0)
    exact_f = wts.sum(dim=0)
    covered = exact_f > 1e-3 * float(exact_f.max())
    n_probes = n_renders * N_PROBES
    sigma = hutchinson_rel_std([wts], n_probes)
    noise = {
        "report_only": True,
        "sigma_rel": sigma,
        "n_probes": n_probes,
        "tol_rel": tol_rel,
        "false_fail_probability_normal": false_fail_probability(sigma, tol_rel),
        "formula": (
            "sigma_rel = sqrt(sum_views 2 (||W W^T||_F^2 - sum_p (sum_i w_ip^2)^2) / n_probes) / S; "
            "false-fail = erfc(tol_rel / (sqrt(2) sigma_rel))"
        ),
    }
    if log is not None:
        log(
            f"toy check, report only (not judged): sigma_rel {sigma:.4g} for {n_probes} probes, "
            f"false-fail probability of the {tol_rel:g} rule under a normal approximation "
            f"{noise['false_fail_probability_normal']:.3g}"
        )
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    est = torch.zeros(n, dtype=torch.float64, device=device)
    formula = torch.zeros(n, dtype=torch.float64, device=device)
    f_rel, formula_rel = 0.0, 0.0
    for _ in range(n_renders):
        s, f, vis, _, probes = probe_view(render_fn, n, device, gen, return_probes=True)
        r = probes[..., :N_PROBES].double().reshape(-1, N_PROBES)  # [P, 16]
        s_formula = (wts.t() @ r).pow(2).mean(dim=1)
        est += s.double()
        formula += s_formula
        rel = (s.double() - s_formula).abs()[covered] / s_formula[covered].clamp_min(
            1e-30
        )
        formula_rel = max(formula_rel, float(rel.max()))
        f_rel = max(
            f_rel,
            float(((f.double() - exact_f).abs()[covered] / exact_f[covered]).max()),
        )
    est /= n_renders
    total_rel = float((est.sum() - exact_s.sum()).abs() / exact_s.sum())
    per = ((est - exact_s).abs() / exact_s.clamp_min(1e-30))[covered]
    return {
        "scene_fixture": TOY_SCENE_FIXTURE,
        "scene_sha256": TOY_SCENE_SHA256,
        "n_splats": n,
        "n_probes": n_probes,
        "image": [TOY_WIDTH, TOY_HEIGHT],
        "n_covering": int(covered.sum()),
        "n_visible": int(vis.sum()),
        "overlap_fraction": float(
            ((wts > 1e-3).sum(1) >= 2).double().sum()
            / ((wts > 1e-3).sum(1) >= 1).double().sum().clamp_min(1)
        ),
        "noise_diagnostic": noise,
        "sum_exact": float(exact_s.sum()),
        "sum_estimate": float(est.sum()),
        "total_rel_err": total_rel,
        "per_splat_rel_err_median": float(per.median()) if len(per) else float("nan"),
        "per_splat_rel_err_p90": float(per.quantile(0.9)) if len(per) else float("nan"),
        "per_splat_rel_err_max": float(per.max()) if len(per) else float("nan"),
        "formula_max_rel_err": formula_rel,
        "f_max_rel_err": f_rel,
        "pass": bool(
            total_rel < tol_rel and formula_rel < tol_exact and f_rel < tol_exact
        ),
    }

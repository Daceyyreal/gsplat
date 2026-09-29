"""E3r's GPU-memory check (asked for before E3r runs; HANDOFF, "E3r notebook"): what the code allocates, in bytes,
at the splat counts of train and bicycle, tied to the peaks E3q and E3p measured. Computed, not guessed:

- **Tile instances.** C3DGS's rasterizer (``submodules/diff-gaussian-rasterization``, the INRIA one) sizes its
  binning buffers by the number of (splat, 16 x 16 tile) pairs a view touches. ``tile_instances`` computes that
  number per view with the rasterizer's own formulas (``forward.cu``: ``in_frustum``'s z <= 0.2 cull,
  ``computeCov2D`` with the 1.3 tan-FoV clamp and the 0.3 low-pass, ``my_radius = ceil(3 sqrt(lambda_max))``,
  ``ndc2Pix``, ``getRect`` with ``BLOCK_X = BLOCK_Y = 16``) on the pinned INRIA checkpoint and its
  ``cameras.json``, over the train views that C3DGS's sensitivity pass renders.
- **C3DGS's allocations** in the compress path that scale with the splat count (``c3dgs_budget``), each with the
  source line it comes from, summed at the backward pass of the sensitivity computation's worst view, the phase
  that holds the most per-splat state (``calc_importance``; E3q's train peak was the same with and without
  fine-tuning). The tie: the same sum at train's size against E3q's measured 4,550,748,160 bytes.
- **E3r's own allocations** (``e3r_budget``): the 16 x 16 metric (544 bytes per splat, float32, packed), the GN
  pass (tied to E3p's measured bicycle GN-pass peak plus the extra width), the cross-validation and the injected run,
  as the code at ``290e62f5`` holds them and as ``harness`` / ``injection`` mitigations would.

    python bench/gn/e3r_memory.py --inria_root <dir with train/ and bicycle/ members>

The INRIA members are fetched with ``e3p_inria.fetch_scene`` (pinned, CRC-checked); they are not in the repository,
so ``tile_instances`` needs them and the rest reads only committed files.
"""

import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(REPO, "kaggle")):
    if p not in sys.path:
        sys.path.insert(0, p)

T4_BYTES = 15_636_037_632  # E3p's and E3q's gn3*_env.json: torch's total_memory of the T4
GB = 1e9
BLOCK = 16
N_SPLATS = {"train": 1_026_508, "bicycle": 6_131_954}
IMAGE_SIZE = {"train": (980, 545), "bicycle": (1237, 822)}  # protocol ii's resolution (E3p): C3DGS's loaded images


def load_scene(inria_dir: str):
    import e3p_inria as ei

    s, info = ei.read_inria_ply(os.path.join(inria_dir, "point_cloud.ply"))
    cams = ei.load_cameras_json(os.path.join(inria_dir, "cameras.json"))
    return s, cams


def split(cams: List[Dict]):
    """INRIA's eval split (``readColmapSceneInfo``, llffhold 8 over the sorted image names)."""
    names = sorted(c["img_name"] for c in cams)
    test = {n for i, n in enumerate(names) if i % 8 == 0}
    return [c for c in cams if c["img_name"] not in test], [c for c in cams if c["img_name"] in test]


def cov3d(s: Dict[str, torch.Tensor]) -> torch.Tensor:
    """``computeCov3D``: ``Sigma = R S S^T R^T`` from exp(scale) and the normalized quaternion (r, x, y, z)."""
    q = torch.nn.functional.normalize(s["quats"].float(), dim=-1)
    r, x, y, z = q.unbind(-1)
    R = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
                     2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
                     2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)
    M = R * torch.exp(s["scales"].float())[:, None, :]
    return M @ M.transpose(1, 2)


def tile_instances(s: Dict[str, torch.Tensor], cam: Dict, W: int, H: int, Sigma: torch.Tensor, chunk: int = 1 << 20,
                   seen: torch.Tensor = None) -> Dict:
    """The rasterizer's ``num_rendered`` for one view (``forward.cu`` ``preprocessCUDA`` and ``getRect``); ``seen``
    (bool per splat) is or-ed with the splats that touch a tile in this view."""
    C2W_R = torch.tensor(cam["rotation"], dtype=torch.float32)  # cameras.json: camera-to-world rotation
    pos = torch.tensor(cam["position"], dtype=torch.float32)  # and camera centre
    tanx, tany = cam["width"] / (2.0 * cam["fx"]), cam["height"] / (2.0 * cam["fy"])
    fx, fy = W / (2 * tanx), H / (2 * tany)  # rasterize_points / rasterizer_impl: focal = size / (2 tan)
    gx, gy = (W + BLOCK - 1) // BLOCK, (H + BLOCK - 1) // BLOCK
    total, visible, n = 0, 0, s["means"].shape[0]
    for a in range(0, n, chunk):
        p = s["means"][a:a + chunk].float()
        t = (p - pos) @ C2W_R  # the view-space point, W2C = [R^T, -R^T c]
        keep = t[:, 2] > 0.2  # in_frustum
        kept_idx = keep.nonzero(as_tuple=True)[0] + a
        t, Sg = t[keep], Sigma[a:a + chunk][keep]
        tz = t[:, 2]
        limx, limy = 1.3 * tanx, 1.3 * tany
        tx = torch.clamp(t[:, 0] / tz, -limx, limx) * tz
        ty = torch.clamp(t[:, 1] / tz, -limy, limy) * tz
        J = torch.zeros(t.shape[0], 2, 3)
        J[:, 0, 0], J[:, 0, 2] = fx / tz, -(fx * tx) / (tz * tz)
        J[:, 1, 1], J[:, 1, 2] = fy / tz, -(fy * ty) / (tz * tz)
        T = J @ C2W_R.T  # J W with W the world-to-view rotation
        c2 = T @ Sg @ T.transpose(1, 2)
        a00, a01, a11 = c2[:, 0, 0] + 0.3, c2[:, 0, 1], c2[:, 1, 1] + 0.3
        det = a00 * a11 - a01 * a01
        mid = 0.5 * (a00 + a11)
        root = torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
        radius = torch.ceil(3.0 * torch.sqrt(torch.maximum(mid + root, mid - root)))
        # the projected centre: full_proj = view @ projection (znear 0.01, zfar 100), ndc = hom / w
        px = ((t[:, 0] / (t[:, 2] * tanx) + 1.0) * W - 1.0) * 0.5
        py = ((t[:, 1] / (t[:, 2] * tany) + 1.0) * H - 1.0) * 0.5
        x0 = torch.clamp(((px - radius) / BLOCK).trunc(), 0, gx)
        y0 = torch.clamp(((py - radius) / BLOCK).trunc(), 0, gy)
        x1 = torch.clamp(((px + radius + BLOCK - 1) / BLOCK).trunc(), 0, gx)
        y1 = torch.clamp(((py + radius + BLOCK - 1) / BLOCK).trunc(), 0, gy)
        tiles = ((x1 - x0) * (y1 - y0)) * (det != 0)
        total += int(tiles.sum().item())
        visible += int((tiles > 0).sum().item())
        if seen is not None:
            seen[kept_idx[tiles > 0]] = True
    return {"img_name": cam["img_name"], "num_rendered": total, "visible": visible}


def scene_tiles(inria_dir: str, scene: str) -> Dict:
    s, cams = load_scene(inria_dir)
    train, test = split(cams)
    W, H = IMAGE_SIZE[scene]
    Sigma = cov3d(s)
    seen = torch.zeros(s["means"].shape[0], dtype=torch.bool)
    per = [tile_instances(s, c, W, H, Sigma, seen=seen) for c in train]
    worst = max(per, key=lambda r: r["num_rendered"])
    return {"scene": scene, "n_splats": int(s["means"].shape[0]), "n_train_views": len(train), "n_test_views": len(test),
            "n_views": len(cams), "image_size": [W, H], "cameras_json_size": [cams[0]["width"], cams[0]["height"]],
            "max_num_rendered": worst["num_rendered"], "worst_view": worst["img_name"],
            "mean_num_rendered": sum(r["num_rendered"] for r in per) / len(per),
            "max_visible": max(r["visible"] for r in per),
            # a splat touching no tile in any train view has zero colour sensitivity, so C3DGS prunes it: an upper
            # bound on the splats it keeps, hence on the colour-quantized ones that need the metric
            "seen_in_any_train_view": int(seen.sum())}


# ------------------------------------------------------------------------------ C3DGS's own peak
# Bytes per splat at the backward pass of one view of calc_importance (compress.py:40-84), with the line that
# allocates each. All float32 unless noted. "persistent" lives through the whole loop; "view" lives during that
# view's forward and backward.
C3DGS_PER_SPLAT = [
    ("persistent", "model parameters: xyz 3, features_dc 3, features_rest 45, opacity 1, scaling 3, scaling_factor 1, "
                   "rotation 4 (scene/gaussian_model.py load_ply, device cuda)", 60 * 4),
    ("persistent", "features_dc.grad + features_rest.grad, accumulated over the views (compress.py:58, 71-72)", 48 * 4),
    ("persistent", "_xyz.grad and _opacity.grad: the rasterizer's dL_dmeans3D and dL_dopacity, accumulated "
                   "(means3D and opacities require grad in render; compress.py never clears them)", (3 + 1) * 4),
    ("persistent", "scaling = scaling_qa(scaling_activation(_scaling)) (compress.py:43-45)", 3 * 4),
    ("persistent", "cov3d, requires_grad, and cov3d.grad (compress.py:46-48, 55)", 2 * 6 * 4),
    ("persistent", "scaling_factor (compress.py:49-51)", 1 * 4),
    ("view", "cov3d_scaled = cov3d * scaling_factor^2, and the saved square (compress.py:62)", (6 + 1) * 4),
    ("view", "screenspace_points: zeros_like, + 0, its retained grad (gaussian_renderer/__init__.py:40-45)", 3 * 3 * 4),
    ("view", "get_xyz: FakeQuantizationHalf, the half and the float copy (scene/gaussian_model.py:177-178, 813-815)", 3 * 2 + 3 * 4),
    ("view", "get_opacity: sigmoid, FakeQuantize output and its bool mask (gaussian_model.py:196-198)", 4 + 4 + 1),
    ("view", "get_features: FakeQuantize outputs of dc and rest, their bool masks, the cat (gaussian_model.py:181-188)",
     48 * 4 + 48 + 48 * 4),
    ("view", "rasterizer: radii int32 (rasterize_points.cu:70)", 4),
    ("view", "rasterizer: GeometryState (rasterizer_impl.cu:155-170: depths 4, clamped 3, radii 4, means2D 8, cov3D 24, "
             "conic 16, rgb 12, tiles 4, offsets 4)", 79),
    ("view", "backward: dL_dmeans3D 3, dL_dmeans2D 3, dL_dcolors 3, dL_dconic 4, dL_dopacity 1, dL_dcov3D 6, dL_dsh 48, "
             "dL_dscales 3, dL_drotations 4 (rasterize_points.cu:153-161)", 75 * 4),
    ("view", "backward through the fake quantizers and the hooks: grad * mask and grad.abs() for dc, rest and cov3d "
             "(compress.py:53-55)", 2 * (48 + 6) * 4),
]
BINNING_PER_INSTANCE = 8 + 8 + 4 + 4 + 12  # keys and values, sorted and unsorted (rasterizer_impl.cu:181-194), and
# CUB's radix-sort alternate buffers (keys 8 + values 4) in its temp storage
IMAGE_STATE_PER_PIXEL = 4 + 4 + 8 + 3 * 4  # accum_alpha, n_contrib, ranges; the output colour


def c3dgs_budget(scene: str, n_splats: int, n_images: int, num_rendered: int, data_device: str = "cuda") -> Dict:
    W, H = IMAGE_SIZE[scene]
    per_splat = sum(b for _k, _w, b in C3DGS_PER_SPLAT)
    parts = {"per_splat_bytes": per_splat, "per_splat_total": per_splat * n_splats,
             "images": n_images * W * H * 3 * 4 if data_device == "cuda" else 0,  # scene/cameras.py: float32 on data_device
             "binning": num_rendered * BINNING_PER_INSTANCE, "image_state": W * H * IMAGE_STATE_PER_PIXEL}
    parts["total"] = sum(v for k, v in parts.items() if k not in ("per_splat_bytes",))
    return parts


# ------------------------------------------------------------------------------ E3r's own steps
M16_BYTES = 136 * 4  # the packed 16 x 16 metric, float32, as gm.GNAccumulator stores it and the cache holds it
ACC16_BYTES = M16_BYTES + 8 + 16 * 4 + 8 + 4  # GNAccumulator: M, F (f64), c3_abs [16], s_sum (f64), n_vis (i32)
ACC15_BYTES = 120 * 4 + 8 + 15 * 4 + 8 + 4
RUNNER_SPLAT_BYTES = (3 + 4 + 3 + 1 + 3 + 45) * 4  # the runner's splats: means, quats, scales, opacities, sh0, shN
COLOUR_BYTES = 48 * 4  # one [N, 48] float32 colour tensor
UPDATE_CHUNK_BYTES = 262144 * (136 * 8 + 256 * 8 + 48 * 8)  # update_centroids' chunk: M f64, its unpack f64, x f64


def e3p_measured(scene: str = "bicycle") -> Dict:
    """E3p's committed step peaks (kaggle/gn_e3p/gn3p/): the GN pass (15 x 15) and the dMSE steps, allocated bytes."""
    m = json.load(open(os.path.join(REPO, "kaggle", "gn_e3p", "gn3p", f"gn3p_meta_{scene}.json")))
    st = m["steps"]
    peak = lambda pre: max(s["cuda_peak"]["allocated"] for s in st if s["name"].startswith(pre))  # noqa: E731
    return {"gn_pass": peak("gn_pass_full"), "dmse": peak("dmse_"), "gn_vq": peak("gn_vq_"),
            "runner": peak("build_runner"), "M15_bytes": m["gn"]["full"]["M_bytes"], "n_splats": m["n_splats"],
            "reserved_max": max(s.get("cuda_peak", {}).get("reserved", 0) for s in st)}


def e3r_budget(n_splats: int, n_vq: int, c3dgs_state: int, mitigated: bool) -> Dict:
    """Device bytes of E3r's steps on a scene: the GN pass, the harness's cross-validation (GN-VQ and its scoring)
    and the injected C3DGS run's colour step. ``n_vq`` splats need the metric in this host (the colour-quantized
    ones; the GN pass still covers every splat). ``mitigated``: one floored metric copy on the device (floored in
    place from a private copy, the unfloored objective read from the host copy), the reference colours kept on the
    host and the metric buffer released during the scoring renders."""
    e = e3p_measured("bicycle")
    render = e["dmse"] - (e["runner"] + e["M15_bytes"] + 2 * (45 * 4) * e["n_splats"] + RUNNER_SPLAT_BYTES * e["n_splats"])
    render = max(render, 0) * n_splats / e["n_splats"]
    gn_pass = e["gn_pass"] * n_splats / e["n_splats"] + (ACC16_BYTES - ACC15_BYTES) * n_splats
    m = M16_BYTES * n_vq
    x = COLOUR_BYTES * n_vq
    runner = RUNNER_SPLAT_BYTES * n_splats
    col = COLOUR_BYTES * n_splats
    if not mitigated:
        cv_vq = runner + col + col + x + m + m + UPDATE_CHUNK_BYTES  # base, ref split, x, M_even, floored clone, chunk
        cv_score = runner + col + col + x + m + col + col + render  # + variant clone and split, renders
        injected = c3dgs_state + m + m + UPDATE_CHUNK_BYTES  # M_sub, floored clone (and the lifted check's clone)
    else:
        cv_vq = runner + col + x + m + UPDATE_CHUNK_BYTES
        cv_score = runner + col + x + col + render
        injected = c3dgs_state + m + UPDATE_CHUNK_BYTES
    return {"gn_pass": gn_pass, "cv_gn_vq": cv_vq, "cv_scoring": cv_score, "injected_colour_step": injected,
            "render_temporaries": render, "metric_on_device": m, "M16_bytes_per_splat": M16_BYTES}


E3Q_TRAIN_PEAK = 4_550_748_160  # E3q attempt 2, c3dgs_ft0's peak allocated bytes (gn3q_results_train.csv)
N_IMAGES = {"train": 301, "bicycle": 194}  # every view is loaded (train and test), E3p's camera counts


def report(tiles: Dict) -> Dict:
    """The tie on train, the projections to bicycle, and E3r's steps (bytes)."""
    rows = {r["config"]: r for r in csv.DictReader(open(os.path.join(REPO, "kaggle", "gn_e3q", "attempt2", "gn3q",
                                                                      "gn3q_results_train.csv"), newline=""))}
    measured = int(rows["c3dgs_ft0"]["peak_allocated_bytes"])
    assert measured == E3Q_TRAIN_PEAK and int(rows["c3dgs_ft5000"]["peak_allocated_bytes"]) - measured < 1_000_000
    out = {"train_tie": {}, "bicycle": {}, "e3r": {}}
    tr = c3dgs_budget("train", N_SPLATS["train"], N_IMAGES["train"], tiles["train"]["max_num_rendered"])
    residual = measured - tr["total"]
    out["train_tie"] = {**tr, "measured": measured, "explained_fraction": tr["total"] / measured, "residual": residual,
                        "residual_per_splat": residual / N_SPLATS["train"]}
    for dd in ("cuda", "cpu"):
        b = c3dgs_budget("bicycle", N_SPLATS["bicycle"], N_IMAGES["bicycle"], tiles["bicycle"]["max_num_rendered"], dd)
        out["bicycle"][dd] = {**b, "code_only": b["total"],
                              "with_train_residual_per_splat": b["total"] + residual / N_SPLATS["train"] * N_SPLATS["bicycle"]}
    for scene in ("train", "bicycle"):
        n = N_SPLATS[scene]
        # C3DGS's state at its colour VQ (compress_color): parameters (after pruning, <= n), images, the colour
        # and covariance importances [n, 48] and [n, 6], the colour features [n, 48] and the quantizer input [n_vq, 48]
        state = (60 * 4 + 48 * 4 + 6 * 4 + 48 * 4 + 48 * 4) * n
        imgs = N_IMAGES[scene] * IMAGE_SIZE[scene][0] * IMAGE_SIZE[scene][1] * 12
        for mit in (False, True):
            for dd, im in (("cuda", imgs), ("cpu", 0)):
                out["e3r"][f"{scene}_{'mitigated' if mit else 'as_coded'}_images_{dd}"] = e3r_budget(n, n, state + im, mit)
    out["m16"] = {"bytes_per_splat": M16_BYTES, "accumulator_bytes_per_splat": ACC16_BYTES,
                  "bicycle_all_splats": M16_BYTES * N_SPLATS["bicycle"],
                  "bicycle_accumulator": ACC16_BYTES * N_SPLATS["bicycle"]}
    out["t4_bytes"] = T4_BYTES
    out["e3p_bicycle_measured"] = e3p_measured("bicycle")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inria_root", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {"tiles": {}, "c3dgs": {}}
    for scene in ("train", "bicycle"):
        d = os.path.join(a.inria_root, scene)
        if not os.path.exists(os.path.join(d, "point_cloud.ply")):
            print(f"{scene}: no members under {d}")
            continue
        t = scene_tiles(d, scene)
        res["tiles"][scene] = t
        print(json.dumps(t))
    if len(res["tiles"]) == 2:
        res["report"] = report(res["tiles"])
        print(json.dumps(res["report"], indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

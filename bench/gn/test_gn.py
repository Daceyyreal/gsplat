"""CPU tests for bench/gn (E0). The CUDA checks run in the notebook's smoke-test cell.

    python -m pytest bench/gn/test_gn.py -q
"""

import json
import math
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # repo root, for gsplat
import diagnostics as gd  # noqa: E402
import g0  # noqa: E402
import gn_metric as gm  # noqa: E402
import sh_basis as sb  # noqa: E402
import toy_render as tr  # noqa: E402


def _classic_3dgs(dirs, coeffs):
    """The usual 3DGS SH_C0..SH_C3 formulas (an independent writing of the same basis)."""
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396,
    ]
    C3 = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435,
    ]
    d = torch.nn.functional.normalize(dirs, dim=-1)
    x, y, z = d[:, 0:1], d[:, 1:2], d[:, 2:3]
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    c = coeffs
    r = C0 * c[:, 0] - C1 * y * c[:, 1] + C1 * z * c[:, 2] - C1 * x * c[:, 3]
    r = (
        r
        + C2[0] * xy * c[:, 4]
        + C2[1] * yz * c[:, 5]
        + C2[2] * (2 * zz - xx - yy) * c[:, 6]
    )
    r = r + C2[3] * xz * c[:, 7] + C2[4] * (xx - yy) * c[:, 8]
    r = r + C3[0] * y * (3 * xx - yy) * c[:, 9] + C3[1] * xy * z * c[:, 10]
    r = (
        r
        + C3[2] * y * (4 * zz - xx - yy) * c[:, 11]
        + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * c[:, 12]
    )
    r = r + C3[4] * x * (4 * zz - xx - yy) * c[:, 13] + C3[5] * z * (xx - yy) * c[:, 14]
    r = r + C3[6] * x * (xx - 3 * yy) * c[:, 15]
    return r


# ---------------------------------------------------------------------------- SH basis


def test_sh_basis_matches_gsplat_reference():
    r = sb.reference_check(n=20000, seed=1)
    assert r["pass"], r


def test_sh_basis_matches_classic_3dgs_formulas():
    g = torch.Generator().manual_seed(2)
    dirs = torch.randn(5000, 3, generator=g, dtype=torch.float64)
    coeffs = torch.randn(5000, 16, 3, generator=g, dtype=torch.float64)
    assert torch.allclose(
        sb.eval_sh(coeffs, dirs), _classic_3dgs(dirs, coeffs), atol=1e-10
    )


def test_sh_lower_degrees_are_prefixes():
    dirs = torch.randn(100, 3, dtype=torch.float64)
    full = sb.sh_basis(dirs, 3)
    for deg in range(3):
        assert torch.equal(sb.sh_basis(dirs, deg), full[:, : (deg + 1) ** 2])
    assert torch.equal(sb.shn_basis(dirs), full[:, 1:])


def test_camera_positions_match_inverse():
    q, _ = torch.linalg.qr(torch.randn(4, 3, 3, dtype=torch.float64))
    vm = torch.eye(4, dtype=torch.float64).repeat(4, 1, 1)
    vm[:, :3, :3] = q
    vm[:, :3, 3] = torch.randn(4, 3, dtype=torch.float64)
    assert torch.allclose(
        sb.camera_positions(vm), torch.linalg.inv(vm)[:, :3, 3], atol=1e-12
    )


# ------------------------------------------------------------------------ packing, GN


def test_pack_unpack_trace_frobenius():
    a = torch.randn(7, 15, 15, dtype=torch.float64)
    a = a + a.transpose(1, 2)
    b = torch.randn(7, 15, 15, dtype=torch.float64)
    b = b + b.transpose(1, 2)
    assert torch.equal(gm.unpack(gm.pack(a)), a)
    assert torch.allclose(
        gm.trace_packed(gm.pack(a)), torch.diagonal(a, dim1=1, dim2=2).sum(-1)
    )
    w = gm.frobenius_weights("cpu", torch.float64)
    assert torch.allclose((w * gm.pack(a) * gm.pack(b)).sum(-1), (a * b).sum((1, 2)))
    y = torch.randn(5, 15, dtype=torch.float64)
    assert torch.allclose(gm.unpack(gm.pack_outer(y)), y[:, :, None] * y[:, None, :])


def test_accumulator_matches_explicit_sum():
    torch.manual_seed(0)
    n, views = 50, 4
    means = torch.randn(n, 3)
    coeffs = torch.randn(n, 16, 3) * 0.3
    acc = gm.GNAccumulator(n, "cpu", chunk=7)
    M_ref = torch.zeros(n, 15, 15, dtype=torch.float64)
    F_ref = torch.zeros(n, dtype=torch.float64)
    c3_ref = torch.zeros(n, 15, dtype=torch.float64)
    neg = tot = 0
    for v in range(views):
        s, f = torch.rand(n), torch.randn(n)
        vis = torch.rand(n) > 0.3
        campos = torch.randn(3) * 5
        acc.add_view(s, f, vis, means, campos, coeffs, n_pixels=100)
        for i in range(n):
            if not vis[i]:
                continue
            basis = sb.sh_basis((means[i] - campos)[None].double(), 3)[0]
            y = basis[1:]
            M_ref[i] += float(s[i]) * torch.outer(y, y)
            F_ref[i] += float(f[i])
            c3_ref[i] += (float(f[i]) * y).abs()
            col = (basis[:, None] * coeffs[i].double()).sum(0) + 0.5
            neg += int((col < 0).sum())
            tot += 3
    res = acc.result()
    assert torch.allclose(gm.unpack(res["M_packed"].double()), M_ref, atol=1e-5)
    assert torch.allclose(res["F"], F_ref, atol=1e-5)
    assert torch.allclose(res["c3dgs"].double(), (c3_ref / views).amax(-1), atol=1e-5)
    assert res["clamp_neg"] == neg and res["clamp_total"] == tot
    assert res["total_pixels"] == views * 100 and res["n_views"] == views


def test_toy_exactness_on_cpu_renderer():
    r = gm.toy_exactness(render=tr.render_bruteforce, device="cpu", seed=0)
    assert r["pass"], r
    assert r["n_probes"] == 64 and r["n_splats"] <= 256
    assert (
        r["overlap_fraction"] > 0.5
    )  # the toy blends splats, it is not a set of isolated blobs


def test_e2e_exactness_on_cpu_renderer():
    r = gd.e2e_exactness(render=tr.render_bruteforce, device="cpu")
    ex = r["non_overlapping"]
    assert r["pass"] and ex["preconditions_ok"] and ex["rel_err"] <= 1e-4, ex
    assert ex["max_splats_per_pixel"] == 1 and ex["overlap_fraction"] == 0.0
    assert ex["n_splats"] == 48 and ex["n_visible_per_view"] == [48, 48]
    assert 0 < ex["sh_plus_half_min"] and ex["sh_plus_half_max"] < 1
    assert 0 < ex["render_min_covered"] and ex["render_max"] < 1
    assert ex["measured_clamped"] == ex["measured_raw"]  # nothing to clamp
    assert r["amplitude"] == 0.1 and r["n_views"] == 2
    # the overlapping variants are reported, not asserted on; they do blend splats
    for key in ("overlapping", "overlapping_shared_delta"):
        ov = r[key]
        assert ov["max_splats_per_pixel"] >= 2 and not ov["preconditions_ok"]
        assert math.isfinite(ov["ratio_raw"]) and math.isfinite(ov["cross_diagonal"])


def test_e2e_exactness_detects_a_render_mismatch():
    """An SH render 0.1% brighter than the GN pass's render fails the 1e-4 check."""

    def skewed(act, colors, *args, sh_degree=None):
        img, info = tr.render_bruteforce(act, colors, *args, sh_degree=sh_degree)
        return (img * 1.001 if sh_degree is not None else img), info

    r = gd.e2e_exactness(render=skewed, device="cpu")
    assert not r["pass"] and r["non_overlapping"]["preconditions_ok"]
    assert r["non_overlapping"]["rel_err"] > 1e-4


def test_selftest_cpu_writes_json_and_stops_on_a_failed_check(tmp_path, monkeypatch):
    import selftest

    path = tmp_path / "gn_selftest.json"
    selftest.main(["--device", "cpu", "--out", str(path)])
    out = json.load(open(path))
    assert out["pass"] and out["device"] == "cpu"
    assert all(out[k]["pass"] for k in selftest.VALIDITY)
    assert selftest.VALIDITY == ("sh_basis", "toy_exactness", "e2e_exactness")
    assert out["lifted_random"]["criterion_version"] == gd.LIFTED_CHECK_VERSION
    # a failed end-to-end check exits non-zero (the notebook's sh() then stops the run)
    monkeypatch.setattr(gm, "toy_exactness", lambda **k: {"pass": True})
    monkeypatch.setattr(selftest, "lifted_random", lambda device: {})
    monkeypatch.setattr(gd, "e2e_exactness", lambda **k: {"pass": False})
    with pytest.raises(SystemExit, match="e2e_exactness"):
        selftest.main(["--device", "cpu", "--out", str(path)])
    assert json.load(open(path))["pass"] is False


def test_compute_gn_f_is_exact_footprint_and_cache_roundtrip(tmp_path):
    splats = gm.toy_scene(64, seed=3)
    cam = gm.toy_camera()
    views = [
        cam,
        {
            **cam,
            "camtoworld": torch.tensor(
                [[1.0, 0, 0, 0.1], [0, 1, 0, -0.1], [0, 0, 1, 0.2], [0, 0, 0, 1]]
            ),
        },
    ]
    settings = gm.RenderSettings(
        False, "classic", "pinhole", False, False, 0.01, 1e10, 3
    )
    res = gm.compute_gn(
        splats, views, settings, seed=0, render=tr.render_bruteforce, log=None
    )
    act = gm.activated(splats)
    exact_f = torch.zeros(64, dtype=torch.float64)
    for v in views:
        with torch.no_grad():
            img, _ = tr.render_bruteforce(
                act,
                torch.eye(64),
                v["camtoworld"],
                v["K"],
                v["width"],
                v["height"],
                settings,
            )
        exact_f += img.double().sum((0, 1))
    assert torch.allclose(res["F"], exact_f, rtol=1e-5, atol=1e-8)
    assert (
        res["n_views"] == 2 and res["total_pixels"] == 2 * gm.TOY_WIDTH * gm.TOY_HEIGHT
    )
    key = gm.cache_key("sha", settings, 2, 0)
    gm.save_cache(str(tmp_path / "c.pt"), res, key)
    back = gm.load_cache(str(tmp_path / "c.pt"), key, "cpu")
    assert torch.equal(back["M_packed"], res["M_packed"]) and back["n_views"] == 2
    assert gm.load_cache(str(tmp_path / "c.pt"), key + "x", "cpu") is None


# ------------------------------------------------------------------------ diagnostics


def test_eigen_stats():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(40, 15, 6, generator=g, dtype=torch.float64)
    m = a @ a.transpose(1, 2)  # rank 6
    packed = gm.pack(m).float()
    packed[3] = 0
    st = gd.eigen_stats(packed, chunk=16)
    lam = np.clip(np.linalg.eigvalsh(gm.unpack(packed.double()).numpy()), 0, None)
    tr_ = lam.sum(-1)
    ok = tr_ > 0
    lam, tr_ = lam[ok], tr_[ok]
    assert np.allclose(st["trace"].numpy()[ok], tr_, rtol=1e-6)
    assert np.allclose(st["pr"].numpy()[ok], tr_**2 / (lam**2).sum(-1), rtol=1e-6)
    assert np.allclose(st["top1"].numpy()[ok], lam[:, -1] / tr_, rtol=1e-6)
    assert math.isnan(float(st["pr"][3])) and float(st["pr"][ok].max()) <= 6.0 + 1e-6
    summary, hist = gd.spectrum_tables(st, "toy")
    assert {r["weighting"] for r in summary} == {"unweighted", "trace_weighted"}
    assert all(r["n_splats"] == 39 for r in summary)
    per_metric = {}
    for r in hist:
        per_metric.setdefault(r["metric"], 0.0)
        per_metric[r["metric"]] += r["count_fraction"]
    assert all(abs(v - 1.0) < 1e-9 for v in per_metric.values())


def test_rank_and_spearman():
    x = np.array([3.0, 1.0, 2.0, 2.0, 5.0])
    assert np.allclose(gd.rank_average(x), [4.0, 1.0, 2.5, 2.5, 5.0])
    a = np.random.default_rng(0).normal(size=1000)
    assert abs(gd.spearman(a, a**3) - 1.0) < 1e-12
    assert abs(gd.spearman(a, -a) + 1.0) < 1e-12
    ties = np.zeros(10)
    assert math.isnan(gd.spearman(ties, a[:10]))


def test_predicted_dmse_matches_loop():
    g = torch.Generator().manual_seed(1)
    n = 30
    a = torch.randn(n, 15, 4, generator=g)
    M = gm.pack(a @ a.transpose(1, 2))
    delta = torch.randn(n, 15, 3, generator=g)
    ref = sum(
        float(delta[i, :, c] @ gm.unpack(M[i : i + 1])[0] @ delta[i, :, c])
        for i in range(n)
        for c in range(3)
    )
    assert abs(gd.predicted_dmse(M, delta, 1000, chunk=7) - ref / 3000) < 1e-9


def test_measure_dmse_counts_clamp_and_out_of_range():
    ref_img = torch.tensor([[[0.5, 1.2, -0.1], [0.3, 0.4, 0.2]]])  # 1 x 2 pixels
    q_img = torch.tensor([[[0.7, 1.5, -0.3], [0.3, 0.4, 0.2]]])

    def render(view, splats):
        return q_img if splats["shN"].sum() > 0 else ref_img

    out = gd.measure_dmse(render, [{}], {"shN": torch.zeros(1)}, {"q": torch.ones(1)})
    q = out["q"]
    assert q["n_values"] == 6 and q["n_views"] == 1
    assert abs(q["raw"] - (0.04 + 0.09 + 0.04) / 6) < 1e-7
    assert abs(q["clamped"] - 0.04 / 6) < 1e-7
    ref = out["_reference"]
    assert ref["n_pixels"] == 2
    assert np.allclose(ref["below_0_fraction_rgb"], [0, 0, 0.5])
    assert np.allclose(ref["above_1_fraction_rgb"], [0, 0.5, 0])
    assert np.allclose(ref["outside_fraction_rgb"], [0, 0.5, 0.5])


def _problem(n=2000, k=256, seed=0):
    """Random shN-like data in the [N, 45] layout, with M_i of every rank 0..15 (rank-deficient
    and zero included) and a codebook near the data."""
    g = torch.Generator().manual_seed(seed)
    x3 = torch.randn(n, 15, 3, generator=g) * 0.3 + 0.2
    a = torch.randn(n, 15, 15, generator=g, dtype=torch.float64)
    rank = torch.arange(n) % 16
    a = a * (torch.arange(15)[None, None, :] < rank[:, None, None])
    M = gm.pack(a @ a.transpose(1, 2)).float()
    C3 = (
        x3[torch.randperm(n, generator=g)[:k]]
        + torch.randn(k, 15, 3, generator=g) * 0.05
    )
    return x3.reshape(n, 45), M, C3.reshape(k, 45)


def _brute(x, M, C, chunk=100):
    """Independent float64 direct distances [N, K]."""
    x3, C3 = x.double().view(-1, 15, 3), C.double().view(-1, 15, 3)
    out = []
    for s in range(0, x3.shape[0], chunk):
        diff = C3[None] - x3[s : s + chunk, None]
        mf = gm.unpack(M[s : s + chunk].double())
        out.append(torch.einsum("nkac,nab,nkbc->nk", diff, mf, diff))
    return torch.cat(out)


def test_lifted_identity_and_argmin_achieve_brute_force_minimum():
    x, M, C = _problem()
    d = _brute(x, M, C)
    # the lifted form is exact: d(i, k) = const_i + u_i . v_k
    x3, C3 = x.view(-1, 15, 3), C.view(-1, 15, 3)
    const = torch.einsum(
        "nac,nab,nbc->n", x3.double(), gm.unpack(M.double()), x3.double()
    )
    lifted = const[:, None] + gd.lifted_u(x3, M) @ gd.lifted_v(C3).t()
    assert torch.allclose(lifted, d, rtol=1e-9, atol=1e-9)
    # fp32 argmin: the achieved distance equals the brute-force minimum (ties allowed)
    lab = gd.lifted_argmin(x, M, C, chunk=333)
    achieved = d.gather(1, lab[:, None])[:, 0]
    best = d.min(dim=1).values
    assert torch.all(achieved <= best * (1 + 1e-5) + 1e-12)
    dmin, arg = gd.brute_force_min(x, M, C, chunk=64)
    assert torch.allclose(dmin, best, rtol=1e-12, atol=1e-15)
    # the sample is drawn from all splats; tr(M) = 0 splats are counted and not evaluated
    chk = gd.lifted_check(x, M, C, n_sample=500, seed=1)
    assert chk["pass"] and chk["n_sample"] == 500, chk
    assert chk["n_zero_trace_in_sample"] > 0
    assert chk["n"] + chk["n_zero_trace_in_sample"] == 500
    assert chk["n_excess_over_tol_scale"] == 0
    assert chk["criterion_version"] == gd.LIFTED_CHECK_VERSION
    assert torch.allclose(
        gd.direct_distance(x, M, C, lab), achieved, rtol=1e-12, atol=1e-15
    )


def _near_tie_problem(n_tie=200, n_normal=1000, n_near=16, rank=3, seed=0):
    """fp32 near-ties on rank-deficient M_i. Tie splats have ``M_i = diag(l_1..l_rank, 0, ...)`` and
    share the range coordinates of one of ``n_near`` centroids exactly (d_min = 0); those centroids
    agree with each other to 1e-4 in the range coordinates and differ by O(1) in the null ones. A far
    group puts the codebook mean far away, so the fp32 product cancels large terms and cannot tell the
    near group apart. Normal splats: random full-rank M_i, O(1) away from every centroid."""
    g = torch.Generator().manual_seed(seed)
    base = torch.zeros(15, 3)
    base[:rank] = 20.0
    near = base + torch.zeros(n_near, 15, 3)
    near[:, :rank] += torch.randn(n_near, rank, 3, generator=g) * 1e-4
    near[:, rank:] += torch.randn(n_near, 15 - rank, 3, generator=g)
    far = -base + torch.randn(n_near, 15, 3, generator=g) * 3.0
    mid = torch.randn(32, 15, 3, generator=g) * 3.0
    C3 = torch.cat([near, far, mid])
    x_tie = near[torch.randint(0, n_near, (n_tie,), generator=g)].clone()
    x_tie[:, rank:] = torch.randn(n_tie, 15 - rank, 3, generator=g) * 2
    M_tie = torch.zeros(n_tie, 15, 15)
    idx = torch.arange(rank)
    M_tie[:, idx, idx] = torch.rand(n_tie, rank, generator=g) + 0.5
    x_norm = torch.randn(n_normal, 15, 3, generator=g) * 3.0
    a = torch.randn(n_normal, 15, 15, generator=g) / 4
    x = torch.cat([x_tie, x_norm]).reshape(-1, 45)
    M = gm.pack(torch.cat([M_tie, a @ a.transpose(1, 2)]))
    return x, M, C3.reshape(-1, 45), n_tie


def test_lifted_check_passes_fp32_near_ties_on_rank_deficient_M():
    x, M, C, n_tie = _near_tie_problem()
    lab = gd.lifted_argmin(x, M, C)
    d_min, _ = gd.brute_force_min(x, M, C)
    excess = gd.direct_distance(x, M, C, lab) - d_min
    # fp32 rounding really picks farther centroids at the near-ties, where d_min is exactly 0 ...
    assert bool((d_min[:n_tie] == 0).all())
    assert int((excess[:n_tie] > 0).sum()) > n_tie // 4
    # ... so the Amendment-2 criterion (excess relative to d_min) fails there
    assert float((excess / d_min.clamp_min(1e-300)).max()) > 1e-4
    chk = gd.lifted_check(x, M, C, n_sample=10**6, seed=0)
    assert chk["pass"] and chk["aggregate_pass"] and chk["per_splat_pass"], chk
    assert chk["n"] == x.shape[0] and chk["n_zero_trace_in_sample"] == 0
    assert chk["n_dmin_zero"] >= n_tie and chk["n_dmin_below_1e-3_scale"] >= n_tie
    assert chk["max_excess_over_scale"] <= 1e-4
    assert chk["sum_excess_over_sum_dmin"] <= 1e-4


def test_lifted_check_fails_a_wrong_assignment(monkeypatch):
    x, M, C = _problem(n=600, k=64, seed=7)
    real = gd.lifted_argmin
    monkeypatch.setattr(
        gd,
        "lifted_argmin",
        lambda x_, M_, C_, chunk=2048: (real(x_, M_, C_, chunk) + 1) % C_.shape[0],
    )
    chk = gd.lifted_check(x, M, C, n_sample=600, seed=0)
    assert not chk["pass"], chk
    assert not chk["aggregate_pass"] and not chk["per_splat_pass"]
    assert chk["n_excess_over_tol_scale"] > 0 and chk["max_excess_over_scale"] > 1e-4


def test_lifted_criterion_branches():
    f64 = lambda *v: torch.tensor(v, dtype=torch.float64)  # noqa: E731
    both = gd.lifted_criterion(f64(1e-5, 0.0, 0.0), f64(1.0, 1.0, 0.0), f64(1e4, 1e4, 1e4))
    assert both["pass"] and both["aggregate_pass"] and both["per_splat_pass"]
    # aggregate fails, every splat within its scale bound
    agg = gd.lifted_criterion(f64(0.5, 0.0, 0.0), f64(1.0, 1.0, 0.0), f64(1e4, 1e4, 1e4))
    assert not agg["pass"] and not agg["aggregate_pass"] and agg["per_splat_pass"]
    # one splat over its scale bound, the aggregate within 1e-4
    per = gd.lifted_criterion(f64(0.0, 0.0, 0.5), f64(1e6, 1.0, 0.0), f64(1e4, 1.0, 1.0))
    assert not per["pass"] and per["aggregate_pass"] and not per["per_splat_pass"]
    assert per["n_excess_over_tol_scale"] == 1 and per["max_excess_over_scale"] == 0.5
    # every minimum exactly 0: no excess passes, any excess fails
    assert gd.lifted_criterion(f64(0.0, 0.0), f64(0.0, 0.0), f64(1.0, 1.0))["pass"]
    zero = gd.lifted_criterion(f64(1e-9, 0.0), f64(0.0, 0.0), f64(1.0, 1.0))
    assert not zero["aggregate_pass"] and zero["sum_excess_over_sum_dmin"] == math.inf


def test_lifted_check_version_mismatch_reruns():
    v = gd.LIFTED_CHECK_VERSION
    assert gd.lifted_check_needed(None)
    assert gd.lifted_check_needed({"pass": True})  # an Amendment-2 record has no version
    assert gd.lifted_check_needed({"pass": True, "criterion_version": v - 1})
    assert gd.lifted_check_needed({"pass": False, "criterion_version": v})
    assert not gd.lifted_check_needed({"pass": True, "criterion_version": v})


def test_assign_exact_guard_and_zero_trace():
    x, M, C = _problem(n=600, k=64, seed=2)
    d = _brute(x, M, C)
    zero = gm.trace_packed(M) <= 0
    assert bool(zero.any())
    best_idx = d.argmin(dim=1)
    new, info = gd.assign_exact(x, M, C, best_idx.clone())
    assert torch.equal(new[~zero], best_idx[~zero])  # an optimal current label is kept
    assert info["zero_trace"] == int(zero.sum())
    l2 = torch.cdist(x.double(), C.double()).argmin(dim=1)
    assert torch.equal(new[zero], l2[zero])
    rand = torch.randint(0, 64, (600,), generator=torch.Generator().manual_seed(3))
    new2, _ = gd.assign_exact(x, M, C, rand)
    achieved = d.gather(1, new2[:, None])[:, 0]
    assert torch.all(achieved[~zero] <= d.min(1).values[~zero] * (1 + 1e-5) + 1e-12)


def test_share_in_l2_topk():
    x, M, C = _problem(n=300, k=40, seed=4)
    l2 = torch.cdist(x, C).argmin(dim=1)
    assert gd.share_in_l2_topk(x, C, l2, M, topk=1)["share_all"] == 1.0
    far = torch.cdist(x, C).argmax(dim=1)
    s = gd.share_in_l2_topk(x, C, far, M, topk=1)
    assert s["share_all"] == 0.0 and s["share_trace_gt_0"] == 0.0
    assert gd.share_in_l2_topk(x, C, far, M, topk=40)["share_all"] == 1.0


def test_update_variants_match_closed_form():
    x, M, C = _problem(n=500, k=20, seed=5)
    labels = torch.cdist(x, C).argmin(dim=1)
    labels[labels == 3] = 4  # cluster 3 empty -> keeps its centroid
    for variant in gd.REFINE_VARIANTS:
        new, kept = gd.update_centroids(x, labels, M, C, variant, eps=1e-4)
        assert kept >= 1
        for k in range(C.shape[0]):
            members = labels == k
            A = gm.unpack(M[members].double()).sum(0)
            tr = float(torch.trace(A))
            if tr <= 0:
                assert torch.equal(new[k], C[k])
                continue
            mu = 1e-4 * tr / 15
            b = torch.einsum(
                "nab,nbc->ac",
                gm.unpack(M[members].double()),
                x[members].double().view(-1, 15, 3),
            )
            if variant == "prox":
                b = b + mu * C[k].double().view(15, 3)
            q = torch.linalg.solve(A + mu * torch.eye(15, dtype=torch.float64), b)
            assert torch.allclose(new[k].double().view(15, 3), q, rtol=1e-4, atol=1e-5)
    with pytest.raises(ValueError):
        gd.update_centroids(x, labels, M, C, "shortlist")


def test_update_keeps_q_old_when_trace_of_cluster_M_is_zero():
    x, M, C = _problem(n=500, k=20, seed=8)
    labels = torch.cdist(x, C).argmin(dim=1)
    labels[labels == 3] = 4  # cluster 3: empty
    M = M.clone()
    unseen = labels == 5
    assert bool(unseen.any())
    M[unseen] = 0.0  # cluster 5: every member unseen, tr(M_i) = 0
    zero = [
        k
        for k in range(C.shape[0])
        if float(gm.trace_packed(M[labels == k].double()).sum()) == 0.0
    ]
    assert {3, 5} <= set(zero)
    for variant in gd.REFINE_VARIANTS:
        new, kept = gd.update_centroids(x, labels, M, C, variant, eps=1e-4)
        assert kept == len(zero)
        for k in range(C.shape[0]):
            assert torch.equal(new[k], C[k]) == (k in zero), (variant, k)


def test_gn_refine_variants_log_every_step_and_prox_is_monotone(monkeypatch):
    x, M, C = _problem(n=800, k=32, seed=6)
    labels = torch.cdist(x, C).argmin(dim=1)
    for variant in gd.REFINE_VARIANTS:
        _, _, hist, rises = gd.gn_refine(
            x,
            C,
            labels,
            M,
            total_pixels=1000,
            variant=variant,
            iters=3,
            topk=8,
            log=None,
        )
        assert [h["step"] for h in hist] == ["start"] + ["assign", "update"] * 3
        assert rises == []
        objs = [h["objective"] for h in hist]
        if variant == "prox":
            assert all(b <= a * (1 + 1e-6) for a, b in zip(objs, objs[1:])), objs
        for h in hist[1::2]:
            assert 0.0 <= h["top64_share_all"] <= 1.0
    # an injected rise in the proximal update is recorded, not raised, and the iterations go on
    real = gd.update_centroids
    monkeypatch.setattr(
        gd, "update_centroids", lambda *a, **k: (real(*a, **k)[0] + 1.0, 0)
    )
    logged = []
    _, _, hist, rises = gd.gn_refine(
        x, C, labels, M, total_pixels=1000, variant="prox", iters=2, topk=8, log=logged.append
    )
    assert [h["step"] for h in hist] == ["start"] + ["assign", "update"] * 2
    assert rises and all(r["after"] > r["before"] * (1 + 1e-6) for r in rises)
    assert {(r["iter"], r["step"]) for r in rises} >= {(1, "update"), (2, "update")}
    assert any("invalid" in m for m in logged)
    _, _, _, rises = gd.gn_refine(
        x, C, labels, M, total_pixels=1000, variant="ridge", iters=1, topk=8, log=None
    )
    assert rises == []  # only the proximal variant is held to monotonicity


# ----------------------------------------------------------------------------- G0 rule


def _g0_rows(values, raw_train=None):
    """values[(scene, K, config)] = (P, D_train, D_test) -> result rows. Unclamped = clamped, except
    ``raw_train[(scene, K, config)]`` overrides the unclamped train measurement."""
    raw_train = raw_train or {}
    return [
        {
            "scene": s,
            "config": c,
            "n_clusters": str(k),
            "seed": "0",
            "predicted": p,
            "measured_train_clamped": a,
            "measured_test_clamped": b,
            "measured_train_raw": raw_train.get((s, k, c), a),
            "measured_test_raw": b,
        }
        for (s, k, c), (p, a, b) in values.items()
    ]


def _separated(p_factor=1.1):
    """Every pair separated by far more than 5%, P = p_factor x D_train."""
    base = {"upstream_l1": 1.0, "plain_l2": 1.3, "lloyd_wopa_area": 0.7}
    out = {}
    for s_i, s in enumerate(g0.SCENES):
        for k_i, k in enumerate(g0.K_VALUES):
            f = 1.0 + 0.5 * k_i + 0.2 * s_i
            for c, v in base.items():
                out[(s, k, c)] = (p_factor * v * f, v * f, 1.05 * v * f)
    return out


OK = {"sh_basis": True, "toy_exactness": True, "e2e_exactness": True}


def _book(res, scene, k, config):
    return next(
        b
        for b in res["calibration"]["codebooks"]
        if (b["scene"], b["K"], b["config"]) == (scene, k, config)
    )


def test_g0_pass_and_misordered_pairs_fail():
    v = _separated()
    res = g0.judge_g0(_g0_rows(v), OK)
    assert res["verdict"] == "pass" and "if_inconclusive" not in res
    assert res["clamped"]["n_non_tied"] == 36 and res["clamped"]["n_pairs"] == 36
    assert res["raw"]["outcome"] == "pass"
    # one non-tied pair misordered by P -> fail
    bad = dict(v)
    p, a, b = bad[("bicycle", 16384, "plain_l2")]
    bad[("bicycle", 16384, "plain_l2")] = (0.1, a, b)
    res = g0.judge_g0(_g0_rows(bad), OK)
    assert res["verdict"] == "fail" and res["clamped"]["n_disagree"] >= 1
    # equal P on a non-tied pair counts as misordered -> fail
    eq = dict(v)
    p0 = eq[("garden", 4096, "upstream_l1")][0]
    _, a, b = eq[("garden", 4096, "plain_l2")]
    eq[("garden", 4096, "plain_l2")] = (p0, a, b)
    assert g0.judge_g0(_g0_rows(eq), OK)["verdict"] == "fail"
    # a measured tie (< 5%) with P reversed is exempt
    tie = dict(v)
    p1, a1, b1 = tie[("garden", 65536, "upstream_l1")]
    tie[("garden", 65536, "plain_l2")] = (p1 * 0.9, a1 * 1.03, b1 * 1.03)
    res = g0.judge_g0(_g0_rows(tie), OK)
    assert res["verdict"] == "pass" and res["clamped"]["n_non_tied"] == 34


def test_g0_ratio_is_reported_not_judged():
    v = _separated()
    raw = {("garden", 4096, "plain_l2"): v[("garden", 4096, "plain_l2")][1] * 1.21}
    res = g0.judge_g0(_g0_rows(v, raw), OK)
    cal = res["calibration"]
    assert cal["range"] == [0.5, 2.0] and cal["summary"]["n_codebooks"] == 18
    assert cal["summary"]["all_calibrated_train_clamped"]
    assert cal["summary"]["all_calibrated_train_raw"]
    book = _book(res, "garden", 4096, "plain_l2")
    assert abs(book["ratio_train_clamped"] - 1.1) < 1e-12
    assert abs(book["ratio_train_raw"] - 1.1 / 1.21) < 1e-12
    assert abs(book["ratio_test_clamped"] - 1.1 / 1.05) < 1e-12
    assert abs(book["cross_diagonal_train"] - (1.21 / 1.1 - 1)) < 1e-12
    assert abs(_book(res, "bicycle", 65536, "upstream_l1")["cross_diagonal_train"] - (1 / 1.1 - 1)) < 1e-12
    # every ratio far outside [0.5, 2] (P = 3 D): no codebook calibrated, the verdict still pass
    res = g0.judge_g0(_g0_rows(_separated(p_factor=3.0)), OK)
    assert res["verdict"] == "pass"
    assert res["calibration"]["summary"]["n_calibrated_train_clamped"] == 0
    assert not res["calibration"]["summary"]["all_calibrated_train_raw"]
    # one codebook out of range (its P stays the lowest of its K): flagged, nothing else changes
    one = dict(v)
    _, a, b = one[("garden", 16384, "lloyd_wopa_area")]
    one[("garden", 16384, "lloyd_wopa_area")] = (0.55 * a, a, b)  # calibrated
    res = g0.judge_g0(_g0_rows(one), OK)
    assert _book(res, "garden", 16384, "lloyd_wopa_area")["calibrated_train_clamped"]
    one[("garden", 16384, "lloyd_wopa_area")] = (0.3 * a, a, b)  # not calibrated
    res = g0.judge_g0(_g0_rows(one), OK)
    assert res["verdict"] == "pass"
    assert not _book(res, "garden", 16384, "lloyd_wopa_area")["calibrated_train_clamped"]
    assert res["calibration"]["summary"]["n_calibrated_train_clamped"] == 17
    # the range is inclusive
    assert g0.calibrated(0.5) and g0.calibrated(2.0)
    assert not g0.calibrated(0.4999) and not g0.calibrated(2.0001)
    assert not g0.calibrated(math.inf) and not g0.calibrated(math.nan)


def test_g0_inconclusive_invalid_incomplete_and_tie_boundary():
    near = {}
    for s in g0.SCENES:
        for k in g0.K_VALUES:
            for i, c in enumerate(g0.CONFIG_ORDER):
                d = 1.0 + 0.01 * i  # all pairs tie (< 5%)
                near[(s, k, c)] = (d, d, d)
    # separate one config on garden K=4096 train views only: 2 non-tied pairs, both agree
    near[("garden", 4096, "lloyd_wopa_area")] = (1.5, 1.5, 1.02)
    res = g0.judge_g0(_g0_rows(near), OK)
    assert res["clamped"]["n_non_tied"] == 2 and res["verdict"] == "inconclusive"
    assert res["if_inconclusive"] == g0.IF_INCONCLUSIVE and "E1" in res["if_inconclusive"]
    # a ratio outside [0.5, 2] no longer changes it (under Amendment 2 it was a fail)
    near[("bicycle", 65536, "plain_l2")] = (5.0, 1.01, 1.01)
    res = g0.judge_g0(_g0_rows(near), OK)
    assert res["verdict"] == "inconclusive"
    assert not _book(res, "bicycle", 65536, "plain_l2")["calibrated_train_clamped"]
    # a misordered non-tied pair outranks inconclusive
    near[("garden", 4096, "lloyd_wopa_area")] = (0.5, 1.5, 1.02)
    res = g0.judge_g0(_g0_rows(near), OK)
    assert res["clamped"]["n_non_tied"] == 2 and res["verdict"] == "fail"
    # invalid outranks fail and pass; incomplete outranks invalid
    v = _separated()
    assert (
        g0.judge_g0(_g0_rows(v), {**OK, "render_parity_garden": False})["verdict"]
        == "invalid"
    )
    assert g0.judge_g0(_g0_rows(near), {**OK, "e2e_exactness": False})["verdict"] == "invalid"
    missing = {
        key: val for key, val in v.items() if key != ("bicycle", 4096, "plain_l2")
    }
    res = g0.judge_g0(_g0_rows(missing), OK)
    assert res["verdict"] == "incomplete" and res["missing"] == [
        "bicycle plain_l2 K=4096 seed=0"
    ]
    assert g0.judge_g0(_g0_rows(missing), {})["verdict"] == "incomplete"
    assert not g0.is_tie(1.0, 1.05) and g0.is_tie(1.0, 1.0499) and g0.is_tie(0.0, 0.0)
    assert not g0.is_tie(0.0, 1e-9)


def test_g0_degenerate_cases():
    v = _separated()
    # an empty validity dict: the selftest results were never recorded -> invalid
    res = g0.judge_g0(_g0_rows(v), {})
    assert res["verdict"] == "invalid" and not res["valid"]
    # zero train error for one codebook, the lowest of its K: its pairs are non-tied and ordered
    # by P as by D -> still pass; its ratio is +inf and not calibrated
    z = dict(v)
    p, a, b = z[("garden", 4096, "lloyd_wopa_area")]
    z[("garden", 4096, "lloyd_wopa_area")] = (p, 0.0, b)
    res = g0.judge_g0(_g0_rows(z), OK)
    assert res["verdict"] == "pass" and res["clamped"]["n_non_tied"] == 36
    book = _book(res, "garden", 4096, "lloyd_wopa_area")
    assert book["ratio_train_clamped"] == math.inf and not book["calibrated_train_clamped"]
    assert book["cross_diagonal_train"] == -1.0  # D_raw = 0 < P
    # ... and with P misordered against that zero it fails like any other pair
    z[("garden", 4096, "lloyd_wopa_area")] = (10.0, 0.0, b)
    assert g0.judge_g0(_g0_rows(z), OK)["verdict"] == "fail"
    # zero measured errors everywhere: every pair is a tie -> inconclusive (never pass)
    zeros = {key: (p, 0.0, 0.0) for key, (p, _, _) in v.items()}
    res = g0.judge_g0(_g0_rows(zeros), OK)
    assert res["clamped"]["n_non_tied"] == 0 and res["verdict"] == "inconclusive"
    assert res["calibration"]["summary"]["n_calibrated_train_clamped"] == 0
    # P = 0 and D = 0: ratio and cross/diagonal ratio undefined (NaN), not calibrated
    both = dict(zeros)
    both[("bicycle", 16384, "upstream_l1")] = (0.0, 0.0, 0.0)
    book = _book(g0.judge_g0(_g0_rows(both), OK), "bicycle", 16384, "upstream_l1")
    assert math.isnan(book["ratio_train_clamped"]) and math.isnan(book["cross_diagonal_train"])
    assert not book["calibrated_train_raw"]
    # P = 0 < D: ratio 0 (not calibrated), cross/diagonal ratio +inf
    assert g0.ratio(0.0, 1.0) == 0.0 and g0.cross_diagonal(0.0, 1.0) == math.inf


def test_g0_same_rule_over_other_rungs():
    """E1 re-judges an inconclusive G0 over four non-GN rungs with the same rule."""
    rungs = ("upstream_l1", "plain_l2", "lloyd_wopa_area", "c3dgs")
    base = {"upstream_l1": 1.0, "plain_l2": 1.3, "lloyd_wopa_area": 0.7, "c3dgs": 0.5}
    vals = {
        (s, k, c): (1.1 * d, d, d) for s in g0.SCENES for k in g0.K_VALUES for c, d in base.items()
    }
    res = g0.judge_g0(_g0_rows(vals), OK, configs=rungs)
    assert res["verdict"] == "pass" and res["configs"] == list(rungs)
    assert res["clamped"]["n_pairs"] == 2 * 3 * 6 * 2
    assert res["calibration"]["summary"]["n_codebooks"] == 24

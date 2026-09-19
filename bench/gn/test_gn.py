"""CPU tests for bench/gn (E0). The CUDA checks run in the notebook's smoke-test cell.

    python -m pytest bench/gn/test_gn.py -q
"""

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


def test_measure_dmse_counts_and_clamp():
    ref_img = torch.tensor([[[0.5, 1.2, -0.1]]])
    q_img = torch.tensor([[[0.7, 1.5, -0.3]]])

    def render(view, splats):
        return q_img if splats["shN"].sum() > 0 else ref_img

    out = gd.measure_dmse(render, [{}], {"shN": torch.zeros(1)}, {"q": torch.ones(1)})[
        "q"
    ]
    assert abs(out["raw"] - (0.04 + 0.09 + 0.04) / 3) < 1e-7
    assert abs(out["clamped"] - (0.04 + 0 + 0) / 3) < 1e-7
    assert out["n_values"] == 3 and abs(out["ref_out_of_range_fraction"] - 2 / 3) < 1e-9


def _refine_problem(n=400, k=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 45, generator=g)
    a = torch.randn(n, 15, 5, generator=g) * torch.rand(n, 1, 1, generator=g)
    M = gm.pack(a @ a.transpose(1, 2) / 5)
    M[:5] = 0  # splats never seen in a train view
    C = x[torch.randperm(n, generator=g)[:k]].clone()
    labels = gd.shortlist_l2(x, C, 1)[:, 0]
    return x, M, C, labels


def _brute_mahalanobis(x, M, C):
    Mf = gm.unpack(M.double())
    d = (C.double()[None] - x.double()[:, None]).view(x.shape[0], C.shape[0], 15, 3)
    return torch.einsum("nkac,nab,nkbc->nk", d, Mf, d)


def test_exhaustive_argmin_matches_brute_force():
    x, M, C, _ = _refine_problem()
    keep = gm.trace_packed(M) > 0
    got = gd.exhaustive_argmin(x[keep], M[keep], C, chunk=37)
    costs = _brute_mahalanobis(x[keep], M[keep], C)
    best = costs.min(dim=1).values
    assert torch.allclose(
        costs.gather(1, got[:, None])[:, 0], best, rtol=1e-9, atol=1e-9
    )


def test_shortlist_rerank_update():
    x, M, C, labels = _refine_problem()
    cand = gd.shortlist_l2(x, C, 4)
    d2 = torch.cdist(x.double(), C.double())
    assert torch.equal(cand[:, 0], d2.argmin(1))
    new = gd.rerank(x, C, M, cand, labels, chunk=33)
    costs = _brute_mahalanobis(x, M, C)
    allowed = torch.cat([cand, labels[:, None]], 1)
    best_allowed = costs.gather(1, allowed).min(1).values
    tr_ = gm.trace_packed(M)
    assert torch.allclose(
        costs.gather(1, new[:, None])[:, 0][tr_ > 0], best_allowed[tr_ > 0]
    )
    assert torch.equal(new[tr_ == 0], cand[tr_ == 0, 0])
    C2, kept = gd.update_centroids(x, new, M, C, eps=0.0)
    for k in range(C.shape[0]):
        members = new == k
        A = gm.unpack(M[members].double()).sum(0)
        if float(torch.trace(A)) <= 0:
            assert torch.equal(C2[k], C[k])
            continue
        b = torch.einsum(
            "nab,nbc->ac",
            gm.unpack(M[members].double()),
            x[members].double().view(-1, 15, 3),
        )
        assert torch.allclose(
            C2[k].double().view(15, 3), torch.linalg.solve(A, b), rtol=1e-4, atol=1e-5
        )


def test_gn_refine_objective_does_not_increase():
    x, M, C, labels = _refine_problem(n=600, k=16, seed=3)
    _, _, hist = gd.gn_refine(
        x,
        C,
        labels,
        M,
        total_pixels=1000,
        iters=3,
        topk=4,
        eps=1e-4,
        n_recall=200,
        log=None,
    )
    for h in hist:
        assert h["objective_after_assign"] <= h["objective_start"] * (1 + 1e-9)
        assert h["objective_after_update"] <= h["objective_after_assign"] * (1 + 1e-6)
        assert 0.0 <= h["recall_recall"] <= 1.0 and h["recall_n"] == 200


# ----------------------------------------------------------------------------- G0 rule


def _rows(scene, pred, train, test):
    return [
        {
            "scene": scene,
            "config": c,
            "seed": "0",
            "predicted": p,
            "measured_train_clamped": a,
            "measured_test_clamped": b,
            "measured_train_raw": a,
            "measured_test_raw": b,
        }
        for c, p, a, b in zip(g0.CONFIG_ORDER, pred, train, test)
    ]


def test_g0_verdicts():
    ok = {"sh_basis": True, "toy_exactness": True}
    good = _rows("garden", [2, 3, 1], [2.2, 3.1, 1.3], [2.5, 3.9, 1.4]) + _rows(
        "bicycle", [1, 1.5, 0.8], [1.1, 1.6, 0.7], [1.2, 1.8, 0.9]
    )
    assert g0.judge_g0(good, ok)["verdict"] == "pass"
    swapped_test = good[:3] + _rows(
        "bicycle", [1, 1.5, 0.8], [1.1, 1.6, 0.7], [1.2, 1.1, 0.9]
    )
    assert g0.judge_g0(swapped_test, ok)["verdict"] == "fail"
    bad_ratio = _rows("garden", [2, 3, 1], [2.2, 3.1, 0.4], [2.5, 3.9, 0.45]) + good[3:]
    v = g0.judge_g0(bad_ratio, ok)
    assert (
        v["verdict"] == "fail"
        and not v["per_scene"]["garden"]["clamped"]["ratios_train_in_range"]
    )
    assert (
        g0.judge_g0(good, {**ok, "render_parity_garden": False})["verdict"] == "invalid"
    )
    assert g0.judge_g0(good[:5], ok)["verdict"] == "incomplete"
    tie = _rows("garden", [1, 1, 2], [1, 1, 2], [1, 1, 2]) + good[3:]
    assert g0.judge_g0(tie, ok)["per_scene"]["garden"]["clamped"]["same_order_train"]

"""CPU tests for bench/gn (E0). The CUDA checks run in the notebook's smoke-test cell.

    python -m pytest bench/gn/test_gn.py -q
"""

import argparse
import json
import math
import os
import sys
import warnings

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # repo root, for gsplat
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "kaggle"))  # the jobs
import batched as bl  # noqa: E402
import diagnostics as gd  # noqa: E402
import g0  # noqa: E402
import g1  # noqa: E402
import gn_metric as gm  # noqa: E402
import gn_vq as vq  # noqa: E402
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


# ------------------------------------------------------------- chunked batched linalg


def _psd_batch(n=1000, rank_mod=7, dtype=torch.float64):
    """PSD 15x15 matrices including zero (rank 0) and rank-deficient ones; ``n`` is not a multiple
    of the max batches the tests use."""
    g = torch.Generator().manual_seed(3)
    a = torch.randn(n, 15, 6, generator=g, dtype=dtype)
    rank = torch.arange(n) % rank_mod
    a = a * (torch.arange(6)[None, None, :] < rank[:, None, None])
    return a @ a.transpose(1, 2)


def test_batched_linalg_equals_the_unchunked_call():
    m = _psd_batch()
    n = m.shape[0]
    spd = m + 1e-3 * torch.eye(15, dtype=torch.float64)  # solvable even where m is zero
    rhs = torch.randn(n, 15, 3, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    assert n % 7 and n % 128 and n % 997  # the batch never divides the problem evenly

    def same(a, b):
        # not bitwise: LAPACK's eigenvalues of a near-singular matrix move by ~1e-15 with the
        # batch's internal blocking, so the results agree to precision, not to the last bit
        return torch.allclose(a, b, rtol=1e-12, atol=1e-12)

    for max_batch in (1, 7, 128, 997, 10_000):
        chunked = bl.batched_linalg(torch.linalg.eigvalsh, m, max_batch=max_batch, log=None)
        assert same(chunked, torch.linalg.eigvalsh(m)), max_batch
        got = bl.batched_linalg(torch.linalg.solve, spd, rhs, max_batch=max_batch, log=None)
        assert same(got, torch.linalg.solve(spd, rhs)), max_batch
    for op in (torch.linalg.cholesky, torch.linalg.inv, torch.linalg.eigvalsh):
        assert same(bl.batched_linalg(op, spd, max_batch=97, log=None), op(spd)), op
    # a struct-sequence return type survives chunking, fields and all
    ref = torch.linalg.inv_ex(spd)
    got = bl.batched_linalg(torch.linalg.inv_ex, spd, max_batch=97, log=None)
    assert type(got) is type(ref) and same(got.inverse, ref.inverse)
    assert torch.equal(got.info, ref.info)
    # keyword arguments reach every chunk
    assert same(
        bl.batched_linalg(torch.linalg.cholesky, spd, max_batch=97, upper=True, log=None),
        torch.linalg.cholesky(spd, upper=True),
    )


def test_batched_linalg_halves_the_batch_on_a_backend_error():
    x = torch.arange(30.0).reshape(30, 1)
    calls = []

    def flaky(t):  # a backend that refuses batches above 4, as cuSOLVER refused 65,536
        calls.append(t.shape[0])
        if t.shape[0] > 4:
            raise RuntimeError(
                "cusolver error: CUSOLVER_STATUS_INVALID_VALUE, when calling "
                "`cusolverDnXsyevBatched_bufferSize(...)`"
            )
        return t * 2

    before = len(bl.linalg_fallbacks())
    out = bl.batched_linalg(flaky, x, max_batch=16, log=None)
    assert torch.equal(out, x * 2)
    assert [c for c in calls if c > 4] == [16, 8]  # 16 -> 8 -> 4, then it stays at 4
    assert sum(c for c in calls if c <= 4) == 30
    fell = bl.linalg_fallbacks()[before:]
    assert [f["batch"] for f in fell] == [16, 8]
    assert fell[-1]["retry_batch"] == 4 and "cusolver" in fell[-1]["error"].lower()
    assert fell[-1]["op"] == "flaky"

    def broken(t):  # anything else is a real failure, raised as it is
        raise RuntimeError("linalg.eigh: (Batch element 3): The algorithm failed to converge")

    with pytest.raises(RuntimeError, match="failed to converge"):
        bl.batched_linalg(broken, x, max_batch=16, log=None)


def test_batched_linalg_remembers_the_batch_that_worked():
    """E0 rediscovered the same reduction on every chunk; now the working batch is reused."""
    bl.reset_linalg_state()
    try:
        x = torch.arange(24.0).reshape(24, 1)
        calls = []

        def flaky(t):
            calls.append(t.shape[0])
            if t.shape[0] > 4:
                raise RuntimeError("cusolver error: CUSOLVER_STATUS_INVALID_VALUE")
            return t * 2

        bl.batched_linalg(flaky, x, max_batch=16, log=None)
        assert [c for c in calls if c > 4] == [16, 8]
        assert bl.linalg_working_batches()["flaky"] == 4
        calls.clear()
        bl.batched_linalg(flaky, x, log=None)  # no max_batch: starts at what worked
        assert calls == [4, 4, 4, 4, 4, 4]
        # the eigendecomposition starts at the batch E0 found on a T4; other ops at the default
        assert bl.OP_MAX_BATCH["linalg_eigvalsh"] == 8192
        assert bl.op_max_batch("linalg_eigvalsh") == 8192
        assert bl.op_max_batch("linalg_solve") == bl.LINALG_MAX_BATCH
    finally:
        bl.reset_linalg_state()


def test_batched_linalg_argument_checks():
    m = _psd_batch(n=10)
    with pytest.raises(ValueError, match="at least one"):
        bl.batched_linalg(torch.linalg.eigvalsh)
    with pytest.raises(ValueError, match="batch sizes differ"):
        bl.batched_linalg(torch.linalg.solve, m, m[:5], log=None)
    empty = m[:0]
    assert bl.batched_linalg(torch.linalg.eigvalsh, empty, log=None).shape == (0, 15)
    assert bl.LINALG_MAX_BATCH <= 65535  # below the batch cuSOLVER refused


def test_finite_report():
    m = _psd_batch(n=20).clone()
    r = bl.finite_report(m, "M")
    assert r["finite"] and r["n_nonfinite_rows"] == 0 and r["name"] == "M"
    assert r["n_rows"] == 20 and r["n_nan_entries"] == 0 and r["n_inf_entries"] == 0
    m[3, 0, 0] = float("nan")
    m[7, 2, 1] = float("inf")
    m[7, 1, 2] = float("-inf")
    r = bl.finite_report(m)
    assert not r["finite"] and r["n_nonfinite_rows"] == 2
    assert r["n_nan_entries"] == 1 and r["n_inf_entries"] == 2
    assert r["n_nonfinite_entries"] == 3 and r["first_nonfinite_rows"] == [3, 7]
    flat = bl.finite_report(torch.tensor([1.0, float("nan")]))
    assert not flat["finite"] and flat["n_nonfinite_rows"] == 1


def test_routed_call_sites_go_through_the_helper(monkeypatch):
    """The two batched linalg calls that run at 65,536 on Kaggle: the spectrum and the refines."""
    seen = []
    real = bl.batched_linalg

    def spy(op, *tensors, **kwargs):
        seen.append((getattr(op, "__name__", str(op)), tensors[0].shape[0]))
        return real(op, *tensors, **kwargs)

    monkeypatch.setattr(bl, "batched_linalg", spy)
    M = gm.pack(_psd_batch(n=300).float())
    gd.eigen_stats(M, chunk=128)
    assert seen == [  # torch names its ops linalg_<op>
        ("linalg_eigvalsh", 128),
        ("linalg_eigvalsh", 128),
        ("linalg_eigvalsh", 44),
    ], seen
    seen.clear()
    x, _, C = _problem(n=300, k=20, seed=9)
    labels = torch.cdist(x, C).argmin(dim=1)
    gd.update_centroids(x, labels, M, C, "prox")
    assert [s[0] for s in seen] == ["linalg_solve"] and seen[0][1] <= 20, seen
    # routing the GN pass's own (batch 1) inverse changed no value, so the GN cache stays valid
    monkeypatch.undo()
    c2w = torch.eye(4, dtype=torch.float64)
    c2w[:3, 3] = torch.tensor([0.3, -0.2, 1.5], dtype=torch.float64)
    assert torch.equal(
        gm.viewmat_of(c2w), torch.linalg.inv_ex(c2w.reshape(1, 4, 4)).inverse
    )
    assert gm.CACHE_VERSION == 1  # M is computed exactly as before


def test_toy_exactness_on_cpu_renderer():
    logged = []
    r = gm.toy_exactness(
        render=tr.render_bruteforce, device="cpu", seed=0, log=logged.append
    )
    assert r["pass"], r
    assert r["n_probes"] == 64 and r["n_splats"] == 256
    assert r["scene_fixture"] == "toy_scene_seed0"
    assert r["scene_sha256"] == gm.TOY_SCENE_SHA256
    assert (
        r["overlap_fraction"] > 0.5
    )  # the toy blends splats, it is not a set of isolated blobs
    # report-only noise diagnostic (Amendment 4), logged before the check is judged
    nd = r["noise_diagnostic"]
    assert nd["report_only"] and nd["n_probes"] == 64 and nd["tol_rel"] == 0.05
    assert 0 < nd["sigma_rel"] < 1 and 0 <= nd["false_fail_probability_normal"] <= 1
    assert nd["false_fail_probability_normal"] == gm.false_fail_probability(
        nd["sigma_rel"], 0.05
    )
    assert len(logged) == 1 and "report only" in logged[0]


def test_hutchinson_rel_std_matches_monte_carlo():
    """sigma_rel of Amendment 4 against the spread of simulated probe estimates of S = sum w^2."""
    g = torch.Generator().manual_seed(11)
    views = []
    for _ in range(2):
        w = torch.rand(40, 6, generator=g, dtype=torch.float64)  # [pixels, splats]
        views.append(w * (torch.rand(40, 6, generator=g) < 0.5))  # partial overlap
    n_probes = 4
    sigma = gm.hutchinson_rel_std(views, n_probes)
    total = sum(float(w.pow(2).sum()) for w in views)
    n_draws = 40000
    est = torch.zeros(n_draws, dtype=torch.float64)
    for w in views:
        r = torch.randint(0, 2, (n_draws, n_probes, w.shape[0]), generator=g).double()
        proj = torch.einsum("tcp,pi->tci", r * 2 - 1, w)  # sum_p w_pi r_p, per probe
        est += proj.pow(2).sum(dim=2).mean(dim=1)
    assert abs(float(est.mean()) / total - 1) < 0.01  # unbiased
    mc = float(est.std()) / total
    assert abs(mc / sigma - 1) < 0.03, (mc, sigma)
    # a single view equals the formula written out for it
    w = views[0]
    gram = w.t() @ w
    direct = math.sqrt(
        2 * (float(gram.pow(2).sum()) - float(w.pow(2).sum(1).pow(2).sum())) / n_probes
    ) / float(w.pow(2).sum())
    assert abs(gm.hutchinson_rel_std([w], n_probes) - direct) < 1e-15


def test_false_fail_probability():
    assert gm.false_fail_probability(0.0) == 0.0
    assert abs(gm.false_fail_probability(0.05 / 1.959963984540054) - 0.05) < 1e-9
    assert gm.false_fail_probability(0.01) < gm.false_fail_probability(0.02) < 1


def test_fixtures_match_pinned_hashes_and_a_fresh_cpu_draw():
    """Provenance (Amendment 4): bitwise when the fixture's CPU capability and torch version match
    this machine, otherwise within 1e-5 with the difference reported."""
    for name, pinned, fresh in (
        (gm.TOY_SCENE_FIXTURE, gm.TOY_SCENE_SHA256, gm.toy_scene(256, 0, "cpu")),
        (gd.E2E_SCENE_FIXTURE, gd.E2E_SCENE_SHA256, gd.e2e_fixture_tensors(0)),
    ):
        fx, meta = gm.load_fixture(name, pinned)
        assert meta["sha256"] == pinned and gm.tensors_sha256(fx) == pinned
        assert all(meta[k] for k in ("cpu_capability", "torch_version", "platform"))
        assert {k: list(v.shape) for k, v in fx.items()} == meta["shapes"]
        prov = gm.fixture_provenance(fx, fresh, meta)
        if prov["mode"] == "tolerance":
            warnings.warn(
                f"{name}: drawn under {prov['fixture_cpu_capability']} / torch "
                f"{prov['fixture_torch_version']}, this machine is {prov['cpu_capability']} / "
                f"torch {prov['torch_version']}: compared within {prov['tol_abs']} (max abs "
                f"difference {prov['max_abs_diff']:.3g}), not bitwise"
            )
        assert prov["ok"], prov
    _, toy_meta = gm.load_fixture(gm.TOY_SCENE_FIXTURE, gm.TOY_SCENE_SHA256)
    assert toy_meta["rederived_from_commit"] == "c69ba388"  # toy_noise.py / .json
    assert toy_meta["rederived_sha256"] == gm.TOY_SCENE_SHA256


def test_fixture_provenance_modes():
    fx, meta = gm.load_fixture(gm.TOY_SCENE_FIXTURE, gm.TOY_SCENE_SHA256)
    here = gm.cpu_environment()
    same = {**meta, "cpu_capability": here["cpu_capability"], "torch_version": here["torch_version"]}
    other = {**same, "cpu_capability": "AVX2" if here["cpu_capability"] != "AVX2" else "DEFAULT"}
    exact = gm.fixture_provenance(fx, fx, same)
    assert exact["mode"] == "bitwise" and exact["ok"] and exact["max_abs_diff"] == 0.0
    nudged = {k: v + 1e-6 if k == "means" else v for k, v in fx.items()}
    assert not gm.fixture_provenance(fx, nudged, same)["ok"]  # bitwise: any change fails
    tol = gm.fixture_provenance(fx, nudged, other)
    assert tol["mode"] == "tolerance" and tol["ok"] and 0 < tol["max_abs_diff"] <= 1e-5
    far = {k: v + 1e-4 if k == "means" else v for k, v in fx.items()}
    assert not gm.fixture_provenance(fx, far, other)["ok"]
    fewer = {k: v for k, v in fx.items() if k != "sh0"}
    assert not gm.fixture_provenance(fx, fewer, other)["ok"]
    other_torch = {**same, "torch_version": "0.0.0"}  # a torch mismatch alone also relaxes
    assert gm.fixture_provenance(fx, nudged, other_torch)["mode"] == "tolerance"


def test_fixture_hash_mismatch_stops_before_rendering(tmp_path, monkeypatch):
    import selftest

    calls = []

    def spy(*a, **k):
        calls.append(1)
        return tr.render_bruteforce(*a, **k)

    # a tampered file (one value changed, metadata untouched) fails its hash
    src_npz, src_js = gm.fixture_paths(gm.TOY_SCENE_FIXTURE)
    with np.load(src_npz) as z:
        arrays = {k: z[k].copy() for k in z.files}
    arrays["opacities"][0] += np.float32(1e-3)
    monkeypatch.setattr(gm, "FIXTURE_DIR", str(tmp_path))
    np.savez(os.path.join(tmp_path, gm.TOY_SCENE_FIXTURE + ".npz"), **arrays)
    with open(src_js) as f, open(os.path.join(tmp_path, gm.TOY_SCENE_FIXTURE + ".json"), "w") as g:
        g.write(f.read())
    with pytest.raises(gm.FixtureHashMismatch, match="toy_scene_seed0"):
        gm.toy_exactness(render=spy, device="cpu", log=None)
    assert calls == []
    monkeypatch.undo()
    # a pinned hash that the committed files do not have stops the selftest before any render
    monkeypatch.setattr(gm, "TOY_SCENE_SHA256", "0" * 64)
    with pytest.raises(gm.FixtureHashMismatch):
        gm.toy_exactness(render=spy, device="cpu", log=None)
    path = tmp_path / "gn_selftest.json"
    monkeypatch.setattr(sb, "reference_check", lambda: calls.append("sh") or {"pass": True})
    with pytest.raises(SystemExit, match="FIXTURE HASH MISMATCH"):
        selftest.main(["--device", "cpu", "--out", str(path)])
    out = json.load(open(path))
    assert out["pass"] is False
    assert not out["fixtures"]["toy_scene_seed0"]["ok"]
    assert out["fixtures"]["e2e_scene_seed0"]["ok"]
    assert "toy_exactness" not in out and "sh_basis" not in out
    monkeypatch.setattr(gd, "E2E_SCENE_SHA256", "0" * 64)
    with pytest.raises(gm.FixtureHashMismatch, match="e2e_scene_seed0"):
        gd.e2e_exactness(render=spy, device="cpu")
    assert calls == []  # nothing rendered, not even the SH check ran


def test_e2e_exactness_on_cpu_renderer():
    r = gd.e2e_exactness(render=tr.render_bruteforce, device="cpu")
    ex = r["non_overlapping"]
    assert r["pass"] and r["status"] == "pass" and ex["status"] == "pass", ex
    assert r["scene_sha256"] == gd.E2E_SCENE_SHA256 and "message" not in r
    assert ex["preconditions_ok"] and all(ex["preconditions"].values())
    assert set(ex["preconditions"]) == set(gd.E2E_PRECONDITIONS)
    assert ex["rel_err"] <= 1e-4
    assert ex["max_splats_per_pixel"] == 1 and ex["overlap_fraction"] == 0.0
    assert ex["n_splats"] == 48 and ex["n_visible_per_view"] == [48, 48]
    assert ex["n_with_sole_pixel_per_view"] == [48, 48]
    # the colour read from the render is gsplat's max(SH + 0.5, 0); it matches the basis value
    assert 0 < ex["rendered_colour_min"] and ex["rendered_colour_max"] < 1
    assert abs(ex["rendered_colour_min"] - ex["sh_plus_half_min_basis"]) < 1e-5
    assert abs(ex["rendered_colour_max"] - ex["sh_plus_half_max_basis"]) < 1e-5
    assert 0 < ex["render_min_covered"] and ex["render_max_covered"] < 1
    assert ex["render_max_abs_uncovered"] == 0.0
    assert ex["measured_clamped"] == ex["measured_raw"]  # nothing to clamp
    assert r["amplitude"] == 0.1 and r["n_views"] == 2
    # the overlapping variants are reported, not asserted on; they do blend splats
    for key in ("overlapping", "overlapping_shared_delta"):
        ov = r[key]
        assert ov["status"] == "reported_only" and ov["max_splats_per_pixel"] >= 2
        assert not ov["preconditions_ok"]
        assert math.isfinite(ov["ratio_raw"]) and math.isfinite(ov["cross_diagonal"])


def test_e2e_exactness_detects_a_render_mismatch():
    """An SH render 0.1% brighter than the GN pass's render fails the 1e-4 check."""

    def skewed(act, colors, *args, sh_degree=None):
        img, info = tr.render_bruteforce(act, colors, *args, sh_degree=sh_degree)
        return (img * 1.001 if sh_degree is not None else img), info

    r = gd.e2e_exactness(render=skewed, device="cpu")
    assert not r["pass"] and r["status"] == "mismatch"
    assert r["non_overlapping"]["preconditions_ok"]
    assert r["non_overlapping"]["rel_err"] > 1e-4


def test_e2e_precondition_failure_is_not_a_mismatch():
    import selftest

    def darkened(act, colors, *args, sh_degree=None):
        if sh_degree is not None:  # splat 0's SH + 0.5 < 0: the renderer clamps its colour to 0
            colors = colors.clone()
            colors[0, 0, :] = -5.0
        return tr.render_bruteforce(act, colors, *args, sh_degree=sh_degree)

    r = gd.e2e_exactness(render=darkened, device="cpu")
    ex = r["non_overlapping"]
    assert not r["pass"] and r["status"] == ex["status"] == "preconditions_failed"
    assert not ex["preconditions"]["sh_plus_half_in_0_1"]
    assert not ex["preconditions"]["rendered_in_0_1"]
    assert ex["preconditions"]["one_splat_per_pixel"]
    assert "rel_err" not in ex and "predicted" not in ex  # nothing was compared
    assert "does not apply" in r["message"] and "not a prediction mismatch" in r["message"]
    msg = selftest.failure_message(
        {
            "fixtures": {"x": {"ok": True}},
            "sh_basis": {"pass": True},
            "toy_exactness": {"pass": True},
            "e2e_exactness": r,
        }
    )
    assert "PRECONDITIONS FAILED" in msg and "MISMATCH" not in msg.replace("not a prediction mismatch", "")
    # overlap is a failed precondition too, when the case is judged
    scene, _ = gm.load_fixture(gd.E2E_SCENE_FIXTURE, gd.E2E_SCENE_SHA256)
    delta = scene.pop("delta")
    wide = {**scene, "scales": scene["scales"] + math.log(gd.E2E_OVERLAP_SCALE)}
    settings = gm.RenderSettings(False, "classic", "pinhole", False, False, 0.01, 1e10, 3)
    case = gd.e2e_case(wide, delta, tr.render_bruteforce, settings, gd.e2e_views(), judged=True)
    assert case["status"] == "preconditions_failed"
    assert not case["preconditions"]["one_splat_per_pixel"]


def test_selftest_cpu_writes_json_and_stops_on_a_failed_check(tmp_path, monkeypatch):
    import selftest

    path = tmp_path / "gn_selftest.json"
    selftest.main(["--device", "cpu", "--out", str(path)])
    out = json.load(open(path))
    assert out["pass"] and out["device"] == "cpu"
    assert all(v["ok"] for v in out["fixtures"].values()) and len(out["fixtures"]) == 2
    assert all(out[k]["pass"] for k in selftest.VALIDITY)
    assert selftest.VALIDITY == ("sh_basis", "toy_exactness", "e2e_exactness")
    assert out["toy_exactness"]["noise_diagnostic"]["report_only"]
    assert out["lifted_random"]["criterion_version"] == gd.LIFTED_CHECK_VERSION
    # the engineering scale test of the batched linalg helper (not a G0 validity entry)
    scale = out["linalg_scale"]
    assert selftest.ENGINEERING == ("linalg_scale",)
    assert scale["pass"] and scale["max_batch"] == bl.LINALG_MAX_BATCH
    assert scale["eigvalsh"]["ok"] and scale["eigvalsh"]["n_zero_trace"] > 0
    assert scale["solve"]["ok"] and scale["solve"]["variants"] == list(gd.REFINE_VARIANTS)
    assert scale["fallbacks"] == []  # the default batch was enough
    msg = selftest.failure_message(
        {
            "fixtures": {"x": {"ok": True}},
            "linalg_scale": {"pass": False, "error": "cusolver error: CUSOLVER_STATUS_INVALID_VALUE"},
            "sh_basis": {"pass": True},
            "toy_exactness": {"pass": True},
            "e2e_exactness": {"pass": True},
        }
    )
    assert "linalg_scale" in msg and "not a G0 validity check" in msg
    # a failed end-to-end check exits non-zero (the notebook's sh() then stops the run)
    monkeypatch.setattr(gm, "toy_exactness", lambda **k: {"pass": True})
    monkeypatch.setattr(selftest, "lifted_random", lambda device: {})
    monkeypatch.setattr(
        gd,
        "e2e_exactness",
        lambda **k: {
            "pass": False,
            "status": "mismatch",
            "tol_rel": 1e-4,
            "non_overlapping": {"rel_err": 0.5},
        },
    )
    with pytest.raises(SystemExit, match="e2e_exactness: MISMATCH"):
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
    d64, m64 = delta.double(), gm.unpack(M.double())  # float64 reference (fp32 varies by CPU)
    ref = sum(
        float(d64[i, :, c] @ m64[i] @ d64[i, :, c]) for i in range(n) for c in range(3)
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


# ------------------------------------------------------------------- GN-VQ (Amendment 5)


def _vq_problem(n=2000, k=64, seed=11, narrow=0.05):
    """A warm start whose ridge update wants to leave the codebook's range: the codebook is squeezed
    into [-narrow, narrow] while the data sit around 0.2, so the update pulls every centroid out."""
    x, M, C = _problem(n=n, k=k, seed=seed)
    C = C.clamp(-narrow, narrow)
    labels = torch.cdist(x, C).argmin(dim=1)
    return x, M, C, labels


def test_gn_vq_quantizer_matches_the_codec(tmp_path):
    """gn_vq's quantizer is the codec's: the writer must produce exactly these codes, and the
    decoder exactly these values."""
    import gn_e0_scene as job

    g = torch.Generator().manual_seed(21)
    n, k = 1024, 32  # a square number of splats, as PngCompression needs
    splats = {
        "means": torch.randn(n, 3, generator=g),
        "quats": torch.randn(n, 4, generator=g),
        "scales": torch.randn(n, 3, generator=g) - 3,
        "opacities": torch.randn(n, generator=g),
        "sh0": torch.randn(n, 1, 3, generator=g),
        "shN": torch.randn(n, 15, 3, generator=g) * 0.2,
    }
    centroids = torch.randn(k, 45, generator=g) * 0.3
    labels = torch.randint(0, k, (n,), generator=g)
    wd = job.write_and_decode(str(tmp_path / "out"), splats, centroids, labels)
    codes, rng = vq.quantize_codebook(centroids)
    check = vq.check_writer_codes(str(tmp_path / "out"), codes)
    assert check["equal"] and check["n_differing"] == 0, check
    meta = json.load(open(tmp_path / "out" / "meta.json"))["shN"]
    assert meta["quantization"] == vq.QUANT_BITS == 6
    assert abs(meta["mins"] - rng["mins"]) < 1e-12 and abs(meta["maxs"] - rng["maxs"]) < 1e-12
    deq = vq.dequantize_codebook(codes, rng)
    assert torch.allclose(wd["decoded"]["shN"].reshape(n, 45), deq[labels], atol=0, rtol=0)
    # the step is the codec's float32 arithmetic, so it agrees with a float64 recomputation
    # from the stored range only to float32 precision
    assert abs(rng["step"] / ((rng["maxs"] - rng["mins"]) / 63) - 1) < 1e-6


def test_accept_by_cluster_keeps_the_old_centroid_when_it_is_better():
    x, M, C, labels = _vq_problem(n=400, k=16, seed=12)
    worse = C + 5.0  # far from every member
    kept, rejected = vq.accept_by_cluster(x, labels, M, C, worse)
    assert rejected == C.shape[0] and torch.equal(kept, C)
    # a cluster that improves is taken: move one centroid onto its members' mean
    better = C.clone()
    members = labels == 0
    better[0] = x[members].mean(dim=0)
    kept, rejected = vq.accept_by_cluster(x, labels, M, C, better)
    assert rejected == C.shape[0] - 1 and torch.equal(kept[0], better[0])
    assert torch.equal(kept[1:], C[1:])


def test_gn_vq_clips_to_the_warm_range_and_never_raises_the_objective():
    x, M, C, labels = _vq_problem()
    lo, hi = float(C.min()), float(C.max())
    C_out, labels_out, rep = vq.gn_vq(
        x, C, labels, M, total_pixels=1000, max_iters=4, rel_tol=0.0, log=None
    )
    assert float(C_out.min()) >= lo and float(C_out.max()) <= hi  # the clip holds
    objs = [h["objective"] for h in rep["history"]]
    assert all(b <= a * (1 + 1e-12) for a, b in zip(objs, objs[1:])), objs
    assert rep["iterations"] == 4 and rep["stopped_because"] == "max_iters"
    assert rep["clip"] and rep["final_quantized_assignment"]
    assert rep["warm_start"]["range"] == [lo, hi]
    assert rep["quantizer"]["bits"] == 6 and rep["quantizer"]["step"] > 0
    assert rep["objective_before_quantization"] <= objs[0]
    assert rep["fraction_outside_warm_range_final"] == 0.0
    # the update wanted to leave the range, and the clip caught it
    outs = [h["fraction_outside_warm_range"] for h in rep["history"] if h["step"] == "update"]
    assert max(outs) > 0
    assert rep["clusters_rejected_by_clip_total"] >= 0
    # the top-64 diagnostic runs at iteration 1 only (it costs more than the assignment)
    withshare = [h["iter"] for h in rep["history"] if "share_all" in h]
    assert withshare == [1]


def test_gn_vq_stopping_rule_and_ablations():
    x, M, C, labels = _vq_problem()
    _, _, rep = vq.gn_vq(x, C, labels, M, 1000, max_iters=10, rel_tol=1.0, log=None)
    assert rep["iterations"] == 1 and rep["stopped_because"] == "rel_tol"
    # without the clip, coordinates leave the warm-start range
    lo, hi = float(C.min()), float(C.max())
    C_noclip, _, rep_noclip = vq.gn_vq(
        x, C, labels, M, 1000, max_iters=3, rel_tol=0.0, clip=False, log=None
    )
    assert not rep_noclip["clip"] and rep_noclip["clusters_rejected_by_clip_total"] == 0
    assert rep_noclip["fraction_outside_warm_range_final"] > 0
    assert float(C_noclip.min()) < lo or float(C_noclip.max()) > hi
    # without the final quantized assignment, the labels are the ones the loop ended with
    _, labels_a, rep_a = vq.gn_vq(x, C, labels, M, 1000, max_iters=2, rel_tol=0.0, log=None)
    _, labels_b, rep_b = vq.gn_vq(
        x, C, labels, M, 1000, max_iters=2, rel_tol=0.0,
        final_quantized_assignment=False, log=None,
    )
    assert rep_b["final_assignment_labels_changed_fraction"] == 0.0
    assert rep_a["final_assignment_labels_changed_fraction"] >= 0.0
    assert not torch.equal(labels_a, labels_b) or rep_a["final_assignment_labels_changed_fraction"] == 0.0


def test_gn_vq_ridge_eps_is_a_knob_and_both_quantizer_ranges_are_reported():
    """Amendment 6: two exploratory rows that differ from the pre-registered variant in the ridge
    alone, and the final codebook's own quantizer range logged beside the warm start's."""
    x, M, C, labels = _vq_problem()
    reports = {}
    for eps in (1e-4, 1e-3, 1e-2):
        C_out, _, rep = vq.gn_vq(
            x, C, labels, M, total_pixels=1000, max_iters=3, rel_tol=0.0, eps=eps, log=None
        )
        reports[eps] = (C_out, rep)
        assert rep["ridge_eps"] == eps
        # the report carries both ranges: the final codebook's own, and the warm start's
        warm = rep["warm_start"]["quantizer"]
        final = rep["quantizer"]
        for r in (warm, final):
            assert r["bits"] == 6 and r["levels"] == 63 and r["step"] > 0
            assert abs(r["step"] - (r["maxs"] - r["mins"]) / 63) <= 1e-6 * abs(r["step"])
        assert warm == vq.codec_range(C)  # the warm start's is the warm-start codebook's
        assert final == vq.codec_range(C_out)  # the final one is the codebook that gets written
        # with the clip on, the written range never grows past the warm start's
        assert warm["mins"] <= final["mins"] and final["maxs"] <= warm["maxs"]
    # the default is the pre-registered value, and a bigger ridge really does change the codebook
    assert vq.RIDGE_EPS == 1e-4
    _, _, rep_default = vq.gn_vq(x, C, labels, M, 1000, max_iters=3, rel_tol=0.0, log=None)
    assert rep_default["ridge_eps"] == 1e-4
    assert not torch.equal(reports[1e-4][0], reports[1e-2][0])
    assert not torch.equal(reports[1e-4][0], reports[1e-3][0])
    # The ridge acts in the update, before the clip and the per-cluster acceptance: `q = (A + mu
    # I)^-1 A c` shrinks toward 0 as mu grows. (The finished codebook is not monotone in eps - the
    # clip and the acceptance rule both intervene - so the knob is checked where it applies.)
    norms = {}
    for eps in (1e-4, 1e-3, 1e-2):
        upd, _ = gd.update_centroids(x, labels, M, C, "ridge", eps)
        norms[eps] = float(upd.double().norm())
    assert norms[1e-2] < norms[1e-3] < norms[1e-4], norms


def test_e1_job_configs_match_amendment_6():
    """The job's table of GN-VQ variants: the pre-registered one at the pre-registered ridge, and
    Amendment 6's two rows differing in the ridge alone."""
    import gn_e1_scene as job

    assert "eps" not in job.VQ_CONFIGS["gn_vq"]  # uses --eps, i.e. vq.RIDGE_EPS
    assert job.VQ_CONFIGS["gn_vq_eps1e3"]["eps"] == 1e-3
    assert job.VQ_CONFIGS["gn_vq_eps1e2"]["eps"] == 1e-2
    for name in ("gn_vq_eps1e3", "gn_vq_eps1e2"):
        spec = dict(job.VQ_CONFIGS[name])
        spec.pop("eps")
        assert spec == job.VQ_CONFIGS["gn_vq"], (name, spec)  # identical otherwise
    # every GN-VQ variant but the pre-registered one is seed 0 at G1's K
    args = argparse.Namespace(n_clusters=job.N_CLUSTERS)
    for name in job.VQ_CONFIGS:
        wanted = _wanted_for(job, args, name, seeds=[0, 1, 2], k_values=[4096, 16384, 65536])
        if name == "gn_vq":
            assert (job.N_CLUSTERS, 1) in wanted and (4096, 0) in wanted
        else:
            assert wanted == [(job.N_CLUSTERS, 0)], (name, wanted)
    # the warm-start quantizer columns Amendment 6 adds, beside the final codebook's
    for col in ("quant_mins", "quant_maxs", "quant_step", "warm_quant_mins", "warm_quant_maxs",
                "warm_quant_step", "ridge_eps"):
        assert col in job.COLUMNS, col


def _wanted_for(job, args, name, seeds, k_values):
    """`gn_e1_scene.main`'s `wanted`, which is a closure, re-stated for the test."""
    if name in job.VQ_CONFIGS and name != "gn_vq":
        return [(args.n_clusters, 0)]
    out = [(args.n_clusters, s) for s in seeds]
    if name in ("lloyd_wopa_area", "gn_vq"):
        out += [(k, 0) for k in k_values if k != args.n_clusters]
    return out


# --------------------------------------------------------------------------- G1 rule


def _g1_row(scene, config, seed, psnr, size, k=65536):
    return {
        "scene": scene,
        "config": config,
        "n_clusters": str(k),
        "seed": str(seed),
        "PSNR": psnr,
        "size_bytes": size,
    }


def _g1_rows(dpsnr=0.06, size_ratio=1.0, k=65536):
    """Baseline rows plus GN-VQ rows offset by dpsnr and size_ratio, on both scenes and 3 seeds."""
    rows = []
    for scene in g1.SCENES:
        for seed in g1.SEEDS:
            base = 26.0 + 0.1 * seed
            size = 16_000_000 + 1000 * seed
            rows.append(_g1_row(scene, g1.BASELINE, seed, base, size, k))
            rows.append(
                _g1_row(scene, g1.GNVQ, seed, base + dpsnr, int(size * size_ratio), k)
            )
    return rows


def test_g1_passes_when_the_gain_holds_on_both_scenes():
    res = g1.judge_g1_only(_g1_rows(dpsnr=0.06))
    assert res["verdict"] == "pass" and res["complete"]
    for scene in g1.SCENES:
        s = res["per_scene"][scene]
        assert abs(s["mean_dPSNR"] - 0.06) < 1e-9 and s["n_negative_seeds"] == 0
        assert s["passes"] and s["n_size_violations"] == 0
    assert res["rule"].startswith("PREREG_GN.md G1 with Amendment 5 b")


def test_g1_fails_on_a_small_mean_a_negative_seed_or_an_oversized_seed():
    # the mean gain is below +0.05 dB
    res = g1.judge_g1_only(_g1_rows(dpsnr=0.04))
    assert res["verdict"] == "fail" and not res["per_scene"]["garden"]["mean_gain_ok"]
    assert res["per_scene"]["garden"]["n_negative_seeds"] == 0
    # one seed loses PSNR, although the mean is far above the threshold
    rows = _g1_rows(dpsnr=0.2)
    row = next(r for r in rows if r["config"] == g1.GNVQ and r["scene"] == "bicycle" and r["seed"] == "1")
    row["PSNR"] = 26.0 + 0.1 * 1 - 0.01
    res = g1.judge_g1_only(rows)
    assert res["verdict"] == "fail"
    assert res["per_scene"]["bicycle"]["n_negative_seeds"] == 1
    assert res["per_scene"]["garden"]["passes"] and not res["per_scene"]["bicycle"]["passes"]
    # a seed more than 0.5% larger counts as negative even with a PSNR gain
    rows = _g1_rows(dpsnr=0.2)
    row = next(r for r in rows if r["config"] == g1.GNVQ and r["scene"] == "garden" and r["seed"] == "0")
    row["size_bytes"] = int(16_000_000 * 1.006)
    res = g1.judge_g1_only(rows)
    assert res["verdict"] == "fail"
    seed0 = next(s for s in res["per_scene"]["garden"]["seeds"] if s["seed"] == 0)
    assert seed0["negative"] and not seed0["size_ok"] and seed0["dPSNR"] > 0
    assert res["per_scene"]["garden"]["n_size_violations"] == 1


def test_g1_size_rule_boundary_and_no_credit_for_being_smaller():
    # exactly +0.5% is allowed, a hair more is not
    ok = g1.compare_seed(  # 16,080,000 is exactly +0.5% of 16,000,000
        _g1_row("garden", g1.GNVQ, 0, 26.1, 16_080_000),
        _g1_row("garden", g1.BASELINE, 0, 26.0, 16_000_000),
    )
    assert ok["size_ok"] and not ok["negative"] and not ok["dominates"]
    assert abs(ok["size_ratio"] - 0.005) < 1e-15
    over = g1.compare_seed(
        _g1_row("garden", g1.GNVQ, 0, 26.1, 16_080_001),
        _g1_row("garden", g1.BASELINE, 0, 26.0, 16_000_000),
    )
    assert not over["size_ok"] and over["negative"]
    # much smaller is compared on PSNR alone: no credit for the bytes, and it can still be negative
    small_worse = g1.compare_seed(
        _g1_row("garden", g1.GNVQ, 0, 25.9, 15_000_000),
        _g1_row("garden", g1.BASELINE, 0, 26.0, 16_000_000),
    )
    assert small_worse["size_ok"] and small_worse["negative"] and not small_worse["dominates"]
    small_better = g1.compare_seed(
        _g1_row("garden", g1.GNVQ, 0, 26.2, 15_000_000),
        _g1_row("garden", g1.BASELINE, 0, 26.0, 16_000_000),
    )
    assert small_better["dominates"] and not small_better["negative"]
    assert abs(small_better["size_ratio"] + 0.0625) < 1e-12
    # a smaller-and-better run passes G1, which is the case Amendment 5 b exists for
    res = g1.judge_g1_only(_g1_rows(dpsnr=0.06, size_ratio=0.96))
    assert res["verdict"] == "pass"
    assert all(v["n_dominating_seeds"] == 3 for v in res["per_scene"].values())


def test_g1_incomplete_when_a_row_is_missing():
    rows = [r for r in _g1_rows() if not (r["config"] == g1.GNVQ and r["seed"] == "2")]
    res = g1.judge_g1_only(rows)
    assert res["verdict"] == "incomplete" and not res["complete"]
    assert res["missing"] == [f"{s} {g1.GNVQ} K=65536 seed=2" for s in g1.SCENES]


def test_bd_rate_and_rd_curves():
    psnr = [25.0, 25.5, 26.0]
    ref = [1.0e7, 1.4e7, 2.0e7]
    assert abs(g1.bd_rate(ref, psnr, ref, psnr)) < 1e-9  # a curve against itself
    cheaper = [b * 0.9 for b in ref]  # 10% fewer bytes at every PSNR
    assert abs(g1.bd_rate(ref, psnr, cheaper, psnr) + 10.0) < 1e-6
    assert abs(g1.bd_rate(cheaper, psnr, ref, psnr) - 100 / 0.9 + 100) < 1e-6
    assert math.isnan(g1.bd_rate(ref, [25.0, 25.5, 26.0], ref, [30.0, 30.5, 31.0]))  # no overlap
    rows = []
    for scene in g1.SCENES:
        for i, k in enumerate(g1.K_VALUES):
            rows.append(_g1_row(scene, g1.BASELINE, 0, psnr[i], int(ref[i]), k))
            rows.append(_g1_row(scene, g1.GNVQ, 0, psnr[i], int(cheaper[i]), k))
    rd = g1.rd_curves(rows)
    for scene in g1.SCENES:
        s = rd["scenes"][scene]
        assert s["complete"] and len(s["points"][g1.GNVQ]) == 3
        assert abs(s["bd_rate_vs_lloyd_wopa_area"][g1.GNVQ] + 10.0) < 1e-6
    full = g1.judge_g1(rows + _g1_rows(dpsnr=0.06))
    assert full["verdict"] == "pass" and set(full["secondary"]) == set(g1.SECONDARIES)
    assert all(r["verdict"] == "incomplete" for r in full["secondary"].values())
    assert len(full["dominance"]) == 6


def test_exploratory_rows_never_enter_g1_or_a_secondary_comparison():
    """Amendment 5 e and Amendment 6: the exploratory rows are reported only. Adding them, with
    absurd numbers, changes no verdict, no mean and no rate-distortion point."""
    rows = _g1_rows(dpsnr=0.06)
    before = g1.judge_g1(rows)
    loud = []
    for scene in g1.SCENES:
        for config in g1.EXPLORATORY:
            for k in g1.K_VALUES:
                loud.append(_g1_row(scene, config, 0, 99.0, 1, k))
    after = g1.judge_g1(rows + loud)
    assert after["verdict"] == before["verdict"] == "pass"
    assert after["per_scene"] == before["per_scene"]
    assert after["seeds_compared"] == before["seeds_compared"]
    assert after["secondary"] == before["secondary"]
    assert after["rd"] == before["rd"]
    assert g1.check_rows(rows + loud)["n_rows"] == len(rows) + len(loud)


def test_check_rows_refuses_rows_e1_did_not_produce():
    """G1's input must hold only E1's rows, even though E0's output is attached to the run."""
    rows = _g1_rows(dpsnr=0.06)
    assert g1.check_rows(rows)["n_rows"] == len(rows)
    counts = g1.check_rows(rows)["per_scene"]["garden"]
    assert counts[g1.BASELINE] == len(g1.SEEDS) and counts[g1.GNVQ] == len(g1.SEEDS)
    # an E0 row: E0's own configs and its refine variants are not E1's
    for config in ("upstream_l1", "plain_l2", "gn_refine_ridge", "gn_refine_prox"):
        with pytest.raises(RuntimeError, match="not E1 rows"):
            g1.check_rows(rows + [_g1_row("garden", config, 0, 26.0, 16_000_000)])
    with pytest.raises(RuntimeError, match="not E1 rows"):
        g1.check_rows(rows + [_g1_row("stump", g1.GNVQ, 0, 26.0, 16_000_000)])
    assert set(g1.KNOWN_CONFIGS) == {g1.BASELINE, g1.GNVQ, g1.UNCOMPRESSED} | set(
        g1.SECONDARIES
    ) | set(g1.EXPLORATORY)


def test_e1_job_refuses_a_results_csv_that_is_not_its_own(tmp_path):
    """The other half of the isolation: E1 appends to and resumes from its own CSV only, so an E0
    file under E1's name stops the job before any GPU work instead of reaching G1 or the bundle."""
    import csv as _csv

    import gn_e0_scene as e0job
    import gn_e1_scene as job

    path = tmp_path / "gn1_results_garden.csv"
    job.assert_e1_csv(str(path))  # missing file: nothing to check
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=job.COLUMNS)
        w.writeheader()
    job.assert_e1_csv(str(path))  # E1's own header
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=e0job.COLUMNS)
        w.writeheader()
        w.writerow({k: "" for k in e0job.COLUMNS})
    with pytest.raises(RuntimeError, match="not an E1 result file"):
        job.assert_e1_csv(str(path))
    assert os.path.exists(path)  # it refuses; it deletes nothing


# --------------------------------------------------------------------------- G2 rules (E2, Amendment 7)

import g2  # noqa: E402

_G2_BASE_BYTES = [14_000_000, 14_500_000, 15_200_000, 16_400_000]  # K = 1,024 ... 65,536
_G2_BASE_PSNR = [25.00, 25.12, 25.21, 25.27]


def _g2_curve_rows(scene, config, byte_factor=1.0, dpsnr=0.0):
    """One scene's four rows for a config: the base curve with bytes scaled and PSNR shifted."""
    return [
        _g1_row(scene, config, 0, p + dpsnr, int(round(b * byte_factor)), k)
        for k, b, p in zip(g2.K_VALUES, _G2_BASE_BYTES, _G2_BASE_PSNR)
    ]


def _g2_rows(gnvq=None, trace=None, scenes=g2.HELD_OUT):
    """Rows for every scene in `scenes`: lloyd_wopa_area, lloyd_trace and upstream_l1 on the base curve,
    GN-VQ transformed by gnvq[scene] = (byte_factor, dpsnr) (default: 10% fewer bytes); lloyd_trace by
    trace[scene] when given."""
    rows = []
    for s in scenes:
        rows += _g2_curve_rows(s, g2.BASELINE)
        rows += _g2_curve_rows(s, g2.UPSTREAM)
        rows += _g2_curve_rows(s, g2.SCALAR, *((trace or {}).get(s, (1.0, 0.0))))
        rows += _g2_curve_rows(s, g2.GNVQ, *((gnvq or {}).get(s, (0.9, 0.0))))
    return rows


def test_g2_bd_measures_on_exact_shifts():
    b, p = _G2_BASE_BYTES, _G2_BASE_PSNR
    # same PSNR at 10% fewer bytes: the fitted log-byte curves differ by a constant, so -10% up to
    # the cubic fit's rounding (about 1e-7 percentage points with PSNR near 25)
    assert abs(g2.bd_rate(b, p, [x * 0.9 for x in b], p) - (-10.0)) < 1e-6
    assert abs(g2.bd_rate(b, p, [x * 1.1 for x in b], p) - 10.0) < 1e-6
    # same bytes at +0.5 dB: exactly +0.5 dB BD-PSNR
    assert abs(g2.bd_psnr(b, p, b, [x + 0.5 for x in p]) - 0.5) < 1e-6
    assert abs(g2.bd_psnr(b, p, b, [x - 0.2 for x in p]) - (-0.2)) < 1e-6
    # no PSNR overlap: BD-rate undefined; no byte overlap: BD-PSNR undefined
    assert math.isnan(g2.bd_rate(b, p, b, [x + 1.0 for x in p]))
    assert math.isnan(g2.bd_psnr(b, p, [x * 10 for x in b], p))
    # degree 3 on four points interpolates exactly, unlike E1's degree 2 default
    assert g2.DEGREE == 3


def test_g2_scene_outcomes_and_the_fallbacks():
    rows = _g2_rows({
        "stump": (0.9, 0.0),  # BD-rate -10%: win on BD-rate
        "bonsai": (1.1, 0.0),  # +10%: loss on BD-rate
        "counter": (1.0, 1.0),  # entirely above in PSNR: BD-rate undefined, BD-PSNR +1 dB: win
        "kitchen": (1.0, -1.0),  # entirely below: BD-rate undefined, BD-PSNR -1 dB: loss
        "room": (10.0, 1.0),  # neither overlaps: loss
    })
    out = {s: g2.judge_scene(rows, s) for s in ("stump", "bonsai", "counter", "kitchen", "room")}
    assert out["stump"]["outcome"] == "win" and out["stump"]["decided_by"] == "bd_rate"
    assert abs(out["stump"]["bd_rate"] + 10.0) < 1e-6
    assert out["bonsai"]["outcome"] == "loss" and out["bonsai"]["decided_by"] == "bd_rate"
    assert out["counter"]["outcome"] == "win" and out["counter"]["decided_by"] == "bd_psnr"
    assert math.isnan(out["counter"]["bd_rate"]) and abs(out["counter"]["bd_psnr"] - 1.0) < 1e-6
    assert out["kitchen"]["outcome"] == "loss" and out["kitchen"]["decided_by"] == "bd_psnr"
    assert out["room"]["outcome"] == "loss" and out["room"]["decided_by"] == "neither_defined"
    assert math.isnan(out["room"]["bd_rate"]) and math.isnan(out["room"]["bd_psnr"])
    # a BD-rate of exactly 0 is not below 0
    zero = g2.judge_scene(_g2_rows({"stump": (1.0, 0.0)}, scenes=("stump",)), "stump")
    assert zero["outcome"] == "loss" and abs(zero["bd_rate"]) < 1e-6


def test_g2a_passes_fails_on_breadth_and_fails_on_magnitude():
    res = g2.judge_g2a(_g2_rows())  # 9 wins at -10%
    assert res["verdict"] == "pass" and res["n_wins"] == 9 and res["complete"]
    assert abs(res["mean_bd_rate"] + 10.0) < 1e-6 and res["mean_ok"] and res["n_bd_rate_defined"] == 9
    assert res["rule"].startswith("PREREG_GN.md Amendment 7 f")
    # breadth: 7 wins fails even with a mean below -5% (7 x -10 and 2 x +10 average -5.56%)
    res = g2.judge_g2a(_g2_rows({"train": (1.1, 0.0), "truck": (1.1, 0.0)}))
    assert res["n_wins"] == 7 and not res["wins_ok"] and res["mean_ok"] and res["verdict"] == "fail"
    # 8 wins pass
    res = g2.judge_g2a(_g2_rows({"truck": (1.1, 0.0)}))
    assert res["n_wins"] == 8 and res["mean_ok"] and res["verdict"] == "pass"
    # magnitude: 9 wins at -2% each fail the mean condition
    res = g2.judge_g2a(_g2_rows({s: (0.98, 0.0) for s in g2.HELD_OUT}))
    assert res["n_wins"] == 9 and res["wins_ok"] and not res["mean_ok"] and res["verdict"] == "fail"


def _pts(*bp):
    """Curve points from (bytes, PSNR) pairs; K is only a label here."""
    return [{"K": 1024 * 4 ** i, "bytes": b, "PSNR": p} for i, (b, p) in enumerate(bp)]


def test_g2_mean_substitute_every_branch():
    """Amendment 8 a: a, b and c, the cheapest point, the strict and inclusive comparisons, the tie."""
    base = _pts((14_000_000, 25.00), (14_500_000, 25.12), (15_200_000, 25.21), (16_400_000, 25.27))
    # a: gn_vq entirely above; three of its points reach the baseline's best PSNR at fewer bytes than
    # the baseline's best point (16.4 MB), and the cheapest of them decides
    gn = _pts((14_200_000, 25.40), (14_600_000, 25.45), (15_000_000, 25.50), (16_800_000, 25.60))
    s = g2.mean_substitute(gn, base)
    assert s["source"] == "substitute_a" and s["new_point"]["bytes"] == 14_200_000
    assert s["ref_point"]["bytes"] == 16_400_000
    assert abs(s["value"] - (-(1 - 14_200_000 / 16_400_000) * 100)) < 1e-12 and s["value"] < 0
    # a: PSNR equal to the baseline's best counts (>=); bytes equal to its best point do not (strict)
    gn = _pts((16_400_000, 25.30), (16_500_000, 25.40), (13_000_000, 25.27), (17_000_000, 25.50))
    s = g2.mean_substitute(gn, base)
    assert s["source"] == "substitute_a" and s["new_point"]["bytes"] == 13_000_000
    gn = _pts((16_400_000, 25.30), (16_500_000, 25.40), (16_600_000, 25.45), (17_000_000, 25.50))
    assert g2.mean_substitute(gn, base)["source"] == "substitute_c"  # a fails on bytes, b on PSNR
    # b: the baseline entirely above gn_vq; the cheapest baseline point that reaches gn_vq's best PSNR
    # (24.60 at 15.9 MB) at fewer bytes, against gn_vq's best point
    gn = _pts((13_000_000, 24.30), (14_000_000, 24.45), (15_000_000, 24.55), (15_900_000, 24.60))
    s = g2.mean_substitute(gn, base)
    assert s["source"] == "substitute_b" and s["ref_point"]["bytes"] == 14_000_000
    assert s["new_point"]["bytes"] == 15_900_000
    assert abs(s["value"] - (15_900_000 / 14_000_000 - 1) * 100) < 1e-12 and s["value"] > 0
    # c, the other side: the baseline entirely above, but no baseline point is cheaper than gn_vq's best
    gn = _pts((10_000_000, 24.00), (11_000_000, 24.10), (12_000_000, 24.20), (13_000_000, 24.30))
    assert g2.mean_substitute(gn, base) == {"value": 0.0, "source": "substitute_c", "new_point": None,
                                            "ref_point": None}
    # a tie on the best PSNR goes to fewer bytes: the baseline's best is then the 15.0 MB point
    tie = _pts((14_000_000, 25.00), (15_000_000, 25.27), (15_500_000, 25.10), (16_400_000, 25.27))
    gn = _pts((14_800_000, 25.40), (15_100_000, 25.45), (16_000_000, 25.50), (16_900_000, 25.55))
    s = g2.mean_substitute(gn, tie)
    assert s["ref_point"]["bytes"] == 15_000_000 and s["new_point"]["bytes"] == 14_800_000


def test_g2_mean_substitute_on_e1_garden_is_minus_9_77_percent():
    """Amendment 8's worked example, from E1's committed bundle: garden has no defined BD-rate, and
    gn_vq at K = 4,096 (14,803,113 B) reaches lloyd_wopa_area's best (16,405,132 B)."""
    repo = os.path.dirname(os.path.dirname(HERE))
    rd = json.load(open(os.path.join(repo, "kaggle", "gn_e1", "gn1", "gn1_g1.json")))["rd"]["scenes"]
    garden = rd["garden"]["points"]
    assert math.isnan(rd["garden"]["bd_rate_vs_lloyd_wopa_area"]["gn_vq"])
    s = g2.mean_substitute(garden["gn_vq"], garden["lloyd_wopa_area"])
    assert s["source"] == "substitute_a" and s["new_point"]["K"] == 4096 and s["ref_point"]["K"] == 65536
    assert s["new_point"]["bytes"] == 14_803_113 and s["ref_point"]["bytes"] == 16_405_132
    assert abs(s["value"] - (-(1 - 14_803_113 / 16_405_132) * 100)) < 1e-12
    assert round(s["value"], 2) == -9.77


def test_g2a_mean_over_all_nine_with_substitutes():
    """Amendment 8 a: every held-out scene enters the mean, with its BD-rate or its substitute."""
    # 4 scenes above the baseline in PSNR at the same bytes (substitute a: the 14.0 MB gn_vq point
    # against the baseline's 16.4 MB best), 5 at -6% BD-rate
    sub_a = -(1 - 14_000_000 / 16_400_000) * 100
    gnvq = {s: (1.0, 1.0) for s in g2.HELD_OUT[:4]} | {s: (0.94, 0.0) for s in g2.HELD_OUT[4:]}
    res = g2.judge_g2a(_g2_rows(gnvq))
    assert res["n_wins"] == 9 and res["n_bd_rate_defined"] == 5
    assert res["n_substituted"] == {"substitute_a": 4, "substitute_b": 0, "substitute_c": 0}
    assert abs(res["mean_bd_rate"] - (4 * sub_a + 5 * -6.0) / 9) < 1e-6 and res["verdict"] == "pass"
    for s in g2.HELD_OUT[:4]:
        v = res["per_scene"][s]
        assert v["mean_term_source"] == "substitute_a" and abs(v["mean_term"] - sub_a) < 1e-9
        assert v["decided_by"] == "bd_psnr" and v["outcome"] == "win"  # the win rule is unchanged
    # a losing scene below the baseline enters with substitute b (+17.1%) and can sink the mean:
    # 8 wins at -6% and one such scene average -3.43%, so G2a fails with 8 wins
    sub_b = (16_400_000 / 14_000_000 - 1) * 100
    res = g2.judge_g2a(_g2_rows({s: (0.94, 0.0) for s in g2.HELD_OUT[:8]} | {"truck": (1.0, -1.0)}))
    v = res["per_scene"]["truck"]
    assert v["outcome"] == "loss" and v["mean_term_source"] == "substitute_b"
    assert abs(v["mean_term"] - sub_b) < 1e-9
    assert res["n_wins"] == 8 and abs(res["mean_bd_rate"] - (8 * -6.0 + sub_b) / 9) < 1e-6
    assert not res["mean_ok"] and res["verdict"] == "fail"
    # the 0% fallback: a scene above in PSNR but never cheaper (bytes x 10, no overlap either way) is a
    # loss and enters with 0%: 8 x -6% and 0% average -5.33%, so G2a passes with 8 wins
    res = g2.judge_g2a(_g2_rows({s: (0.94, 0.0) for s in g2.HELD_OUT[:8]} | {"truck": (10.0, 1.0)}))
    v = res["per_scene"]["truck"]
    assert v["outcome"] == "loss" and v["decided_by"] == "neither_defined"
    assert v["mean_term_source"] == "substitute_c" and v["mean_term"] == 0.0
    assert abs(res["mean_bd_rate"] - 8 * -6.0 / 9) < 1e-6 and res["verdict"] == "pass"


def test_g2a_all_undefined_passes_now_and_failed_under_amendment_7():
    """The case Amendment 8 was written for: every held-out scene like E1's garden (GN-VQ entirely
    above). Under Amendment 7 the mean over defined BD-rates did not exist, so G2a failed with 9 of 9
    wins; now each scene enters with substitute a and the mean is defined."""
    res = g2.judge_g2a(_g2_rows({s: (1.0, 1.0) for s in g2.HELD_OUT}))
    assert res["n_wins"] == 9 and res["n_bd_rate_defined"] == 0
    assert res["n_substituted"]["substitute_a"] == 9
    # what Amendment 7 computed: the mean over the scenes with a defined BD-rate - none
    old_defined = [v["bd_rate"] for v in res["per_scene"].values() if not math.isnan(v["bd_rate"])]
    assert old_defined == []
    sub_a = -(1 - 14_000_000 / 16_400_000) * 100
    assert abs(res["mean_bd_rate"] - sub_a) < 1e-9 and res["mean_ok"] and res["verdict"] == "pass"
    # and the mirror image, every scene entirely below: 9 losses, substitute b everywhere
    res = g2.judge_g2a(_g2_rows({s: (1.0, -1.0) for s in g2.HELD_OUT}))
    assert res["n_wins"] == 0 and res["n_substituted"]["substitute_b"] == 9
    assert res["mean_bd_rate"] > 0 and res["verdict"] == "fail"


def test_g2a_boundaries_are_inclusive(monkeypatch):
    """At least 8 wins; the mean at most -5%: both boundaries included."""
    table = {}

    def fake_judge_scene(rows, scene, config, baseline, k_values, seed):
        outcome, term = table[scene]
        return {"scene": scene, "missing": [], "outcome": outcome, "bd_rate": term,
                "bd_psnr": math.nan, "mean_term": term, "mean_term_source": "bd_rate"}

    monkeypatch.setattr(g2, "judge_scene", fake_judge_scene)
    table.update({s: ("win", -5.0) for s in g2.HELD_OUT[:8]} | {g2.HELD_OUT[8]: ("loss", -5.0)})
    res = g2.judge_g2a([])
    assert res["n_wins"] == 8 and res["mean_bd_rate"] == -5.0 and res["verdict"] == "pass"
    table[g2.HELD_OUT[8]] = ("loss", -4.99)
    assert g2.judge_g2a([])["verdict"] == "fail"  # mean -4.998...
    table.update({g2.HELD_OUT[7]: ("loss", -5.0), g2.HELD_OUT[8]: ("loss", -5.0)})
    assert g2.judge_g2a([])["n_wins"] == 7 and g2.judge_g2a([])["verdict"] == "fail"


def test_g2_h2b_mean_uses_the_substitutes_and_upstream_has_none():
    """Amendment 8 a, 'same construction elsewhere': H2b's reported mean takes the substitutes (its
    verdict is still the win count); the upstream_l1 comparison reports per-scene terms, no mean."""
    trace = {s: (1.0, -1.0) for s in g2.HELD_OUT[:3]}  # lloyd_trace below GN-VQ by 1 dB, same bytes
    res = g2.judge_h2b(_g2_rows(trace=trace))
    assert res["n_bd_rate_defined"] == 6 and res["n_substituted"]["substitute_a"] == 3
    assert res["mean_bd_rate"] is not None and "mean_ok" not in res
    assert res["verdict"] == "pass" and res["n_wins"] == 9
    out = g2.judge_e2(_g2_rows(trace=trace))
    up = out["reported"]["vs_upstream_l1"]["stump"]
    assert up["mean_term_source"] == "bd_rate" and "mean_bd_rate" not in out["reported"]


def test_g2_exploratory_rows_dev_only_and_never_read():
    """Amendment 8 b: gn_vq_eps1e4 exists on garden and bicycle only, and no verdict reads it."""
    rows = _g2_rows()
    dev = _g2_rows(scenes=g2.DEV)
    explore = [dict(r, config=g2.EPS1E4, PSNR=r["PSNR"] - 0.5) for r in dev if r["config"] == g2.GNVQ]
    before, after = g2.judge_e2(rows + dev), g2.judge_e2(rows + dev + explore)
    for key in ("g2a", "h2b"):
        assert json.dumps(before[key], sort_keys=True, default=str) == json.dumps(
            after[key], sort_keys=True, default=str)
    ex = after["reported"]["exploratory"]["garden"]
    assert ex[f"{g2.EPS1E4}_vs_{g2.BASELINE}"]["outcome"] in ("win", "loss")
    assert ex[f"{g2.GNVQ}_vs_{g2.EPS1E4}"]["outcome"] == "win"  # eps 1e-2 is 0.5 dB above here
    assert before["reported"]["exploratory"]["garden"][f"{g2.GNVQ}_vs_{g2.EPS1E4}"]["outcome"] == "incomplete"
    assert g2.check_rows(rows + dev + explore)["per_scene"]["garden"][g2.EPS1E4] == 4
    stray = [dict(explore[0], scene="stump")]
    with pytest.raises(RuntimeError, match="exploratory"):
        g2.check_rows(rows + stray)


def test_g2_incomplete_and_the_development_scenes_are_never_read():
    rows = _g2_rows()
    drop = next(r for r in rows if r["scene"] == "flowers" and r["config"] == g2.GNVQ
                and r["n_clusters"] == "16384")
    res = g2.judge_g2a([r for r in rows if r is not drop])
    assert res["verdict"] == "incomplete" and res["missing"] == ["flowers gn_vq K=16384 seed=0"]
    assert res["per_scene"]["flowers"]["outcome"] == "incomplete"
    # a missing lloyd_trace row leaves G2a complete and makes H2b incomplete
    drop_t = next(r for r in rows if r["scene"] == "truck" and r["config"] == g2.SCALAR)
    kept = [r for r in rows if r is not drop_t]
    assert g2.judge_g2a(kept)["verdict"] == "pass" and g2.judge_h2b(kept)["verdict"] == "incomplete"
    # garden and bicycle: absurd rows change neither verdict, and they are reported as development
    dev = _g2_rows({s: (100.0, -5.0) for s in g2.DEV}, scenes=g2.DEV)
    before, after = g2.judge_e2(rows), g2.judge_e2(rows + dev)
    for key in ("g2a", "h2b"):
        assert json.dumps(before[key], sort_keys=True, default=str) == json.dumps(
            after[key], sort_keys=True, default=str)
    assert set(after["reported"]["development"]) == set(g2.DEV)
    assert after["reported"]["development"]["garden"][g2.BASELINE]["outcome"] == "loss"
    assert set(g2.HELD_OUT) & set(g2.DEV) == set() and len(g2.HELD_OUT) == 9


def test_h2b_threshold_and_independence_from_g2a():
    # GN-VQ vs lloyd_trace: lloyd_trace at 5% fewer bytes than the baseline, GN-VQ at 10%: a win
    trace = {s: (0.95, 0.0) for s in g2.HELD_OUT}
    res = g2.judge_h2b(_g2_rows(trace=trace))
    assert res["verdict"] == "pass" and res["n_wins"] == 9 and "mean_ok" not in res
    # lloyd_trace better than GN-VQ on 3 scenes: 6 wins, H2b fails while G2a still passes
    worse = trace | {s: (0.8, 0.0) for s in g2.HELD_OUT[:3]}
    rows = _g2_rows(trace=worse)
    assert g2.judge_h2b(rows)["n_wins"] == 6 and g2.judge_h2b(rows)["verdict"] == "fail"
    assert g2.judge_g2a(rows)["verdict"] == "pass"
    worse = trace | {s: (0.8, 0.0) for s in g2.HELD_OUT[:2]}
    assert g2.judge_h2b(_g2_rows(trace=worse))["verdict"] == "pass"  # 7 wins
    assert g2.judge_e2(rows)["verdict"] == g2.judge_g2a(rows)["verdict"]  # the gate is G2a


def test_g2_reported_extras_and_input_check():
    rows = _g2_rows()
    out = g2.judge_e2(rows)
    eq = out["reported"]["equal_k"]["stump"]
    assert len(eq) == 12  # 4 K x 3 baselines
    one = next(x for x in eq if x["K"] == 4096 and x["baseline"] == g2.BASELINE)
    assert abs(one["size_ratio"] + 0.1) < 1e-6 and one["dPSNR"] == 0.0 and one["dominates"]
    assert out["reported"]["vs_upstream_l1"]["stump"]["outcome"] == "win"
    # only E2's own rows: E1's exploratory and secondary configs, and E0's, are refused
    for config in ("gn_vq_eps1e2", "lloyd_c3dgs", "plain_l2", "gn_refine_ridge"):
        with pytest.raises(RuntimeError, match="not E2 rows"):
            g2.check_rows(rows + [_g1_row("stump", config, 0, 26.0, 16_000_000)])
    with pytest.raises(RuntimeError, match="not E2 rows"):
        g2.check_rows(rows + [_g1_row("playroom", g2.GNVQ, 0, 26.0, 16_000_000)])
    assert g2.check_rows(rows)["n_rows"] == len(rows)


def test_e2_job_constants_columns_and_foreign_csv(tmp_path):
    """Amendment 7's variant and grid as the job's defaults; E1's columns plus the E2 ones; a CSV that
    is not E2's (E1's, for instance) refused before any work."""
    import csv as _csv

    import gn_e1_scene as e1job
    import gn_e2_scene as job

    assert job.VQ_EPS == 1e-2 and job.VQ_MAX_ITERS == 20
    assert job.K_VALUES == "1024,4096,16384,65536" and tuple(job.CONFIGS) == g2.CONFIGS
    assert job.ROW_ORDER[:2] == (g2.BASELINE, g2.GNVQ)  # the G2a pair first at every K
    # Amendment 8 b: one exploratory variant, differing from gn_vq in the ridge alone
    assert job.EXPLORATORY == {g2.EPS1E4: {"eps": 1e-4}} and job.VQ_VARIANTS == (g2.GNVQ, g2.EPS1E4)
    assert g2.EPS1E4 not in job.CONFIGS and g2.EPS1E4 not in job.ROW_ORDER
    assert set(e1job.COLUMNS) < set(job.COLUMNS)
    assert set(job.COLUMNS) - set(e1job.COLUMNS) == {"dataset", "scene_set", "data_factor", "vq_max_iters"}
    assert job.COLUMNS.index("vq_max_iters") == job.COLUMNS.index("vq_iterations") - 1
    assert [job.scene_set(s) for s in ("stump", "truck", "garden")] == ["held_out", "held_out", "development"]
    with pytest.raises(ValueError, match="not an E2 scene"):
        job.scene_set("playroom")
    path = tmp_path / "gn2_results_stump.csv"
    job.assert_e2_csv(str(path))
    with open(path, "w", newline="") as f:
        _csv.DictWriter(f, fieldnames=job.COLUMNS).writeheader()
    job.assert_e2_csv(str(path))
    with open(path, "w", newline="") as f:
        _csv.DictWriter(f, fieldnames=e1job.COLUMNS).writeheader()
    with pytest.raises(RuntimeError, match="not an E2 result file"):
        job.assert_e2_csv(str(path))
    assert os.path.exists(path)


def test_e2_pinned_checkpoints_match_run5_and_amendment_7():
    """The checkpoint sha1s the E2 notebook pins are the ones runs 4-5 measured, and Amendment 7's."""
    import csv as _csv
    import re

    repo = os.path.dirname(os.path.dirname(HERE))
    rows = list(_csv.DictReader(open(os.path.join(repo, "kaggle", "run5", "tilequant", "run5_results.csv"), newline="")))
    run5 = {r["scene"]: r["ckpt_sha1"] for r in rows}
    assert set(run5) == set(g2.SCENES)
    builder = open(os.path.join(repo, "kaggle", "build_gn_e2_bench.py"), encoding="utf-8").read()
    pinned = dict(re.findall(r'"(\w+)": \("(?:tandt|mipnerf360)", "\w+", "([0-9a-f]{40})"\)', builder))
    assert pinned == run5
    amendment = open(os.path.join(repo, "kaggle", "PREREG_GN.md"), encoding="utf-8").read().split("## Amendment 7")[1]
    table = dict(re.findall(r"\| (\w+) \| (?:held-out|development) \| [^|]+ \| `([0-9a-f]{40})` \|", amendment))
    assert table == run5
    # the queue: the held-out scenes first, garden and bicycle last
    order = list(pinned)
    assert set(order[:9]) == set(g2.HELD_OUT) and order[9:] == list(g2.DEV)


# ---------------------------------------------------------------- E2b (Amendment 9) and the BD computation

import bd_sensitivity as bds  # noqa: E402
import e2b  # noqa: E402


def test_bd_scaled_matches_the_exact_cubic_on_all_of_e2s_curves():
    """Amendment 9 a: g2.bd_rate_scaled / g2.bd_psnr_scaled (numpy.polynomial.Polynomial.fit) against
    the 50-digit exact interpolating cubic, on every ordered pair of E2's curves within a scene, to 1e-8
    (percentage points and dB). g2.bd_rate / g2.bd_psnr, which produced E2's recorded values, stay as they
    were: at H2b treehill they are further off, in a machine-dependent way."""
    import itertools

    rows = bds.load_rows()
    n = 0
    for s in g2.SCENES:
        cfgs = list(g2.CONFIGS) + (list(g2.EXPLORATORY) if s in g2.DEV else [])
        for a, b in itertools.permutations(cfgs, 2):
            ref, new = bds.points(rows, s, a), bds.points(rows, s, b)
            args = (ref[0], ref[1], new[0], new[1])
            for fn, exact in ((g2.bd_rate_scaled, bds.exact_bd_rate), (g2.bd_psnr_scaled, bds.exact_bd_psnr)):
                got, want = fn(*args), exact(*args)
                assert math.isnan(got) == math.isnan(want), (s, a, b, fn.__name__)
                if not math.isnan(want):
                    assert abs(got - want) <= 1e-8, (s, a, b, fn.__name__, got, want)
            n += 1
    assert n == 9 * 12 + 2 * 20  # 4 configs on 11 scenes, plus the eps = 1e-4 curve on garden and bicycle
    # undefined exactly where the old functions are, and exact on exact shifts
    b, p = _G2_BASE_BYTES, _G2_BASE_PSNR
    assert math.isnan(g2.bd_rate_scaled(b, p, b, [x + 1.0 for x in p]))
    assert math.isnan(g2.bd_psnr_scaled(b, p, [x * 10 for x in b], p))
    assert abs(g2.bd_rate_scaled(b, p, [x * 0.9 for x in b], p) + 10.0) < 1e-10
    assert abs(g2.bd_psnr_scaled(b, p, b, [x + 0.5 for x in p]) - 0.5) < 1e-12


def test_bd_sensitivity_reproduces_e2_within_tolerance_and_refuses_beyond_it():
    """bench/gn/bd_sensitivity.py: E2's gn2_g2.json reproduces with g2's own code (every categorical field
    identical, values within 2e-3 pp and 1e-6 dB); a value moved beyond the tolerance, or a changed
    outcome, raises. The committed output agrees with a fresh run on everything that is not rounding."""
    import copy

    rows = bds.load_rows()
    bundle = json.load(open(os.path.join(bds.BUNDLE, "gn2_g2.json")))
    out = bds.analyse(rows, bundle)
    assert out["reproduction"]["ok"] and out["reproduction"]["n_comparisons"] == 39
    assert out["sign_disagreement"]["flagged"]["cubic_bundle"] == ["h2b/treehill"]
    assert out["pchip"]["g2a"]["n_wins"] == 8
    assert out["held_out_without_flagged_scenes_post_hoc"]["h2b"]["n_wins"] == 8
    committed = json.load(open(bds.OUT))
    for key in ("sign_disagreement", "held_out_without_flagged_scenes_post_hoc", "pchip", "monotonicity"):
        assert json.dumps(committed[key], sort_keys=True) == json.dumps(out[key], sort_keys=True), key
    for path, delta in ((("h2b", "per_scene", "treehill", "bd_rate"), -0.003),
                        (("g2a", "per_scene", "stump", "bd_psnr"), 2e-6), (("g2a", "mean_bd_rate"), 0.01)):
        t = copy.deepcopy(bundle)
        d = t
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] += delta
        with pytest.raises(AssertionError, match="do not reproduce"):
            bds.analyse(rows, t)
    t = copy.deepcopy(bundle)
    t["g2a"]["per_scene"]["room"]["outcome"] = "loss"
    with pytest.raises(AssertionError, match="outcome"):
        bds.analyse(rows, t)


def test_e2b_floored_metric():
    """M_i + rho tr(M_i) / 15 I: the input itself at rho = 0; otherwise only the diagonal moves, the
    trace scales by 1 + rho, zero-trace splats stay zero, and every eigenvalue rises by rho tr / 15."""
    _, M, _ = _problem(n=300, k=8)
    assert e2b.floored_metric(M, 0.0) is M
    off = gm.TRIU_I != gm.TRIU_J
    for rho in (1e-3, 1e-2, 1e-1):
        F = e2b.floored_metric(M, rho)
        assert F.dtype == M.dtype and F.shape == M.shape and not torch.equal(F, M)
        assert torch.equal(F[:, off], M[:, off])
        tr, trf = gm.trace_packed(M.double()), gm.trace_packed(F.double())
        assert torch.allclose(trf, (1 + rho) * tr, rtol=1e-6, atol=0)
        zero = tr == 0
        assert bool(zero.any()) and torch.equal(F[zero], M[zero])
        lam = torch.linalg.eigvalsh(gm.unpack(M.double()))
        lamf = torch.linalg.eigvalsh(gm.unpack(F.double()))
        assert torch.allclose(lamf - lam, (rho * tr / 15)[:, None].expand_as(lam), rtol=1e-4, atol=1e-5)
    with pytest.raises(ValueError):
        e2b.floored_metric(M, -1e-3)


def test_e2b_floored_metric_goes_through_the_lifted_assignment_exactly():
    """The lifted fp32 assignment on a floored metric achieves the brute-force minimum, and the lifted
    check passes, as for the pure metric (Amendment 9 b.g runs it per floored metric)."""
    x, M, C = _problem()
    F = e2b.floored_metric(M, 1e-1)
    d = _brute(x, F, C)
    lab = gd.lifted_argmin(x, F, C, chunk=333)
    assert torch.all(d.gather(1, lab[:, None])[:, 0] <= d.min(dim=1).values * (1 + 1e-5) + 1e-12)
    assert gd.lifted_check(x, F, C, n_sample=500, seed=1)["pass"]
    # the floor changes which centroid is nearest for some splats: it is not a no-op
    assert not torch.equal(lab, gd.lifted_argmin(x, M, C, chunk=333))


def test_e2b_rho_zero_is_gn_vq_and_report_metrics_changes_nothing():
    x, M, C, labels = _vq_problem()
    kw = dict(total_pixels=1000, max_iters=3, rel_tol=0.0, eps=1e-2, log=None)
    Ca, La, ra = vq.gn_vq(x, C, labels, M, **kw)
    Cb, Lb, rb = vq.gn_vq(x, C, labels, e2b.floored_metric(M, 0.0), report_metrics={"M": (M, 1000)}, **kw)
    assert torch.equal(Ca, Cb) and torch.equal(La, Lb)
    assert [h["objective"] for h in ra["history"]] == [h["objective"] for h in rb["history"]]
    assert "objectives_under" not in ra
    under = rb["objectives_under"]["M"]
    assert under["objective_before_quantization"] == rb["objective_before_quantization"]
    assert under["objective_after_quantization"] == rb["objective_after_quantization"]
    # a floored run reports the unfloored objective below its own (the floor only adds)
    F = e2b.floored_metric(M, 1e-1)
    _, _, rf = vq.gn_vq(x, C, labels, F, report_metrics={"M": (M, 1000)}, **kw)
    assert rf["objectives_under"]["M"]["objective_after_quantization"] < rf["objective_after_quantization"]


def test_e2b_views_labels_and_selection():
    views = list(range(7))
    assert e2b.even_odd(views) == ([0, 2, 4, 6], [1, 3, 5])
    assert [e2b.rho_label(r) for r in e2b.RHOS] == ["0", "1e-3", "1e-2", "1e-1"]
    assert e2b.RHOS == (0.0, 1e-3, 1e-2, 1e-1) and e2b.K_VALUES == (4096, 65536)
    assert e2b.select_rho_cv({0.0: 3.0, 1e-3: 2.0, 1e-2: 2.5, 1e-1: 4.0}) == 1e-3
    assert e2b.select_rho_cv({0.0: 2.0, 1e-3: 2.0, 1e-2: 2.0, 1e-1: 1.0}) == 1e-1
    assert e2b.select_rho_cv({0.0: 1.0, 1e-3: 1.0, 1e-2: 2.0, 1e-1: 1.0}) == 0.0  # exact tie: the smaller rho
    assert e2b.select_rho_cv({0.0: 1.0, 1e-3: 1.0, 1e-2: 2.0}) is None
    assert e2b.select_rho_cv({0.0: 1.0, 1e-3: float("nan"), 1e-2: 2.0, 1e-1: 1.0}) is None


def _e2b_fixture(treehill_psnr=None, garden_psnr=None, odd=None, full_dmse=None, rho0=None, drop=()):
    """E2 rows (lloyd_wopa_area, lloyd_trace, gn_vq) and E2b rows for the four scenes at both K.

    E2: lloyd_trace at 23.30 dB with test dMSE 2e-4, gn_vq at 23.20 dB with test dMSE 1e-4 and
    15,000,000 B. E2b's own rho = 0 full-M row reproduces E2's gn_vq exactly unless `rho0(scene, k)`
    returns overrides. Full-M rows at rho > 0: PSNR from `treehill_psnr` / `garden_psnr` (rho -> PSNR,
    default 23.25), test dMSE from `full_dmse(scene, k, rho)` (default 1.2 x the odd score). `odd` maps
    rho -> odd-view dMSE (default: rho 1e-2 lowest). `drop` removes (scene, config, K, rho) rows."""
    odd = odd or {0.0: 4e-4, 1e-3: 3e-4, 1e-2: 2e-4, 1e-1: 5e-4}
    e2_rows, rows = [], []
    for s in e2b.SCENES:
        for k in e2b.K_VALUES:
            for c, p, d in ((g2.BASELINE, 23.10, 3e-4), (g2.SCALAR, 23.30, 2e-4), (g2.GNVQ, 23.20, 1e-4)):
                e2_rows.append({**_g1_row(s, c, 0, repr(p), "15000000", k), "measured_test_clamped": repr(d)})
            for rho in e2b.RHOS:
                common = {"scene": s, "n_clusters": str(k), "seed": "0", "rho": repr(rho),
                          "m_source": "restored_cache", "warm_start_source": "e2_work_cache"}
                rows.append({**common, "config": e2b.CV, "measured_odd_clamped": repr(odd[rho]),
                             "measured_test_clamped": repr(odd[rho] * 1.1), "size_bytes": "15000000"})
                if rho == 0:
                    full = {"PSNR": repr(23.2), "size_bytes": "15000000", "measured_test_clamped": repr(1e-4)}
                    full.update((rho0 or (lambda s_, k_: {}))(s, k))
                else:
                    table = {"treehill": treehill_psnr, "garden": garden_psnr}.get(s) or {}
                    d = (full_dmse or (lambda s_, k_, r_: None))(s, k, rho)
                    full = {"PSNR": repr(table.get(rho, 23.25)), "size_bytes": "15010000",
                            "measured_test_clamped": repr(d if d is not None else odd[rho] * 1.2)}
                rows.append({**common, "config": e2b.FULL, "train_PSNR": "24.0", **full})
    rows = [r for r in rows if (r["scene"], r["config"], int(r["n_clusters"]), float(r["rho"])) not in set(drop)]
    return e2_rows, rows


def test_e2b_psnr_criterion_every_path():
    """Amendment 9 b.e, as Amendment 10 e keeps it (reported): treehill must beat E2's lloyd_trace at both
    K (strictly), garden must stay within 0.02 dB of E2's gn_vq at both K (inclusive); at rho_cv = 0 the
    codebook is E2's gn_vq row; a missing row is incomplete."""
    e2_rows, rows = _e2b_fixture(treehill_psnr={1e-2: 23.31}, garden_psnr={1e-2: 23.19})
    res = e2b.judge_e2b(rows, e2_rows)
    c = res["criterion_psnr"]
    assert res["verdicts"]["psnr"] == "works" and c["treehill_ok"] and c["garden_ok"]
    assert c["caveat"] == e2b.CROSS_TERM_CAVEAT
    v = res["per_scene"]["treehill"]["65536"]
    assert v["rho_cv"] == 1e-2 and v["rho_cv_label"] == "1e-2" and v["missing"] == []
    assert v["rho_cv_vs_e2"][g2.SCALAR]["dPSNR"] == round(23.31 - 23.30, 9)
    # equal to lloyd_trace is not "higher"
    e2_rows, rows = _e2b_fixture(treehill_psnr={1e-2: 23.30}, garden_psnr={1e-2: 23.19})
    c = e2b.judge_e2b(rows, e2_rows)["criterion_psnr"]
    assert c["verdict"] == "does not work" and not c["treehill_ok"] and c["garden_ok"]
    # garden exactly 0.02 dB below E2's gn_vq passes; a hair more fails
    e2_rows, rows = _e2b_fixture(treehill_psnr={1e-2: 23.31}, garden_psnr={1e-2: 23.18})
    assert e2b.judge_e2b(rows, e2_rows)["criterion_psnr"]["garden_ok"]
    e2_rows, rows = _e2b_fixture(treehill_psnr={1e-2: 23.31}, garden_psnr={1e-2: 23.1799999})
    assert not e2b.judge_e2b(rows, e2_rows)["criterion_psnr"]["garden_ok"]
    # rho_cv = 0: Amendment 9's codebook is E2's gn_vq row, even when E2b's own rho = 0 row differs
    e2_rows, rows = _e2b_fixture(odd={0.0: 1e-4, 1e-3: 3e-4, 1e-2: 2e-4, 1e-1: 5e-4},
                                 rho0=lambda s, k: {"PSNR": repr(23.5)})
    t = e2b.judge_e2b(rows, e2_rows)["per_scene"]["treehill"]["4096"]
    assert t["rho_cv"] == 0.0 and t["rho_cv_codebook"]["source"] == "e2_gn_vq_row"
    assert t["rho_cv_codebook"]["PSNR"] == 23.20
    # a missing CV row: no rho_cv, incomplete
    e2_rows, rows = _e2b_fixture(treehill_psnr={1e-2: 23.31}, drop=[("treehill", e2b.CV, 4096, 1e-3)])
    res = e2b.judge_e2b(rows, e2_rows)
    assert res["verdicts"]["psnr"] == "incomplete" and res["per_scene"]["treehill"]["4096"]["rho_cv"] is None
    assert any("treehill gn_vq_floor_cv K=4096 rho=1e-3" in m for m in res["criterion_psnr"]["missing"])


def test_e2b_fidelity_criterion_every_path():
    """Amendment 10 d: R = test dMSE at rho_cv / E2's lloyd_trace; works if R is below E2b's own rho = 0
    R in >= 5 of the 6 cells of treehill, flowers and stump, and treehill's R at K = 65,536 is below 1.
    Garden control (reported): test dMSE at rho_cv at most 5% above its own rho = 0."""
    better = lambda s, k, r: 0.8e-4 if r == 1e-2 else None  # below own rho = 0 (1e-4) everywhere
    e2_rows, rows = _e2b_fixture(full_dmse=better)
    res = e2b.judge_e2b(rows, e2_rows)
    f = res["criterion_fidelity"]
    assert res["verdicts"]["fidelity"] == "works" and f["n_cells_below_own_rho0"] == 6 and f["n_cells"] == 6
    assert f["treehill_R_at_max_K"] == pytest.approx(0.4) and f["treehill_R_below_1"]
    assert f["cells"]["stump/4096"] == {"rho_cv": 1e-2, "R_rho_cv": pytest.approx(0.4), "R_rho0": pytest.approx(0.5),
                                        "below_own_rho0": True}
    assert "train/4096" not in f["cells"] and set(e2b.FIDELITY_SCENES) == {"treehill", "flowers", "stump"}
    assert res["verdicts"]["garden_control"] is True
    assert f["garden_control"]["65536"]["test_dmse_rho_cv_over_rho0"] == pytest.approx(0.8)
    # the default fixture: R at rho_cv is 1.2, above own rho = 0 (0.5) everywhere and above 1 on treehill
    e2_rows, rows = _e2b_fixture()
    f = e2b.judge_e2b(rows, e2_rows)["criterion_fidelity"]
    assert f["verdict"] == "does not work" and f["n_cells_below_own_rho0"] == 0 and not f["treehill_R_below_1"]
    assert f["garden_control_ok"] is False  # 2.4e-4 is 140% above 1e-4
    # 4 of 6 cells: flowers' two cells no better than its own rho = 0
    e2_rows, rows = _e2b_fixture(full_dmse=lambda s, k, r: (1e-4 if s == "flowers" else 0.8e-4) if r == 1e-2 else None)
    f = e2b.judge_e2b(rows, e2_rows)["criterion_fidelity"]
    assert f["n_cells_below_own_rho0"] == 4 and f["verdict"] == "does not work"
    # 5 of 6 is enough
    e2_rows, rows = _e2b_fixture(full_dmse=lambda s, k, r: (1e-4 if (s, k) == ("flowers", 4096) else 0.8e-4)
                                 if r == 1e-2 else None)
    assert e2b.judge_e2b(rows, e2_rows)["criterion_fidelity"]["verdict"] == "works"
    # 6 of 6 below own rho = 0, but treehill's R at K = 65,536 is not below 1: own rho = 0 at 3e-4 there
    e2_rows, rows = _e2b_fixture(
        full_dmse=lambda s, k, r: (2e-4 if (s, k) == ("treehill", 65536) else 0.8e-4) if r == 1e-2 else None,
        rho0=lambda s, k: {"measured_test_clamped": repr(3e-4)} if (s, k) == ("treehill", 65536) else {})
    f = e2b.judge_e2b(rows, e2_rows)["criterion_fidelity"]
    assert f["n_cells_below_own_rho0"] == 6 and f["treehill_R_at_max_K"] == pytest.approx(1.0)
    assert not f["treehill_R_below_1"] and f["verdict"] == "does not work"
    # rho_cv = 0: R equals its own rho = 0 value, which is not below it; garden then costs 0%
    e2_rows, rows = _e2b_fixture(odd={0.0: 1e-4, 1e-3: 3e-4, 1e-2: 2e-4, 1e-1: 5e-4})
    f = e2b.judge_e2b(rows, e2_rows)["criterion_fidelity"]
    assert f["n_cells_below_own_rho0"] == 0 and f["verdict"] == "does not work" and f["garden_control_ok"]
    # garden control: exactly 5% above passes, a hair more fails; neither changes the fidelity verdict
    for d, ok in ((1.05e-4, True), (1.0500001e-4, False)):
        e2_rows, rows = _e2b_fixture(
            full_dmse=lambda s, k, r, d=d: (d if s == "garden" else 0.8e-4) if r == 1e-2 else None)
        res = e2b.judge_e2b(rows, e2_rows)
        assert res["verdicts"]["garden_control"] is ok and res["verdicts"]["fidelity"] == "works"
    # E2b's own rho = 0 row missing on stump: incomplete; a missing garden row leaves the verdict alone
    e2_rows, rows = _e2b_fixture(full_dmse=better, drop=[("stump", e2b.FULL, 65536, 0.0)])
    assert e2b.judge_e2b(rows, e2_rows)["verdicts"]["fidelity"] == "incomplete"
    e2_rows, rows = _e2b_fixture(full_dmse=better, drop=[("garden", e2b.FULL, 65536, 0.0)])
    res = e2b.judge_e2b(rows, e2_rows)
    assert res["verdicts"]["fidelity"] == "works" and res["verdicts"]["garden_control"] is None


def test_e2b_reproduction_check():
    """Amendment 10 c: E2b's own rho = 0 full-M row against E2's gn_vq row."""
    e2_rows, rows = _e2b_fixture()
    rep = e2b.judge_e2b(rows, e2_rows)["reproduction_rho0"]
    assert set(rep["cells"]) == {f"{s}/{k}" for s in e2b.SCENES for k in e2b.K_VALUES}
    assert all(v["status"] == "identical" for v in rep["cells"].values()) and rep["flagged"] == []
    cases = {
        ("treehill", 4096): ({"PSNR": repr(23.2005)}, "within_tolerance"),  # 5e-4 dB
        ("treehill", 65536): ({"measured_test_clamped": repr(1.0005e-4)}, "within_tolerance"),
        ("flowers", 4096): ({"PSNR": repr(23.21)}, "not_reproduced"),  # 0.01 dB
        ("flowers", 65536): ({"measured_test_clamped": repr(1.01e-4)}, "not_reproduced"),  # 1% dMSE
        ("stump", 4096): ({"size_bytes": "15000001"}, "within_tolerance"),  # bytes alone
        ("stump", 65536): ({"PSNR": repr(23.3), "m_source": "recomputed"}, "inputs_differ"),
        ("garden", 4096): ({"warm_start_source": "recomputed"}, "inputs_differ"),
    }
    e2_rows, rows = _e2b_fixture()
    for r in rows:
        key = (r["scene"], int(r["n_clusters"]))
        if r["config"] == e2b.FULL and float(r["rho"]) == 0 and key in cases:
            r.update(cases[key][0])
    rep = e2b.judge_e2b(rows, e2_rows)["reproduction_rho0"]
    for (s, k), (_over, status) in cases.items():
        assert rep["cells"][f"{s}/{k}"]["status"] == status, (s, k, rep["cells"][f"{s}/{k}"])
    assert rep["cells"]["garden/65536"]["status"] == "identical"
    assert rep["flagged"] == ["flowers/4096", "flowers/65536"]
    assert rep["cells"]["stump/65536"]["dPSNR"] == pytest.approx(0.1)  # still reported
    assert rep["tolerance"] == {"PSNR_db": 1e-3, "test_dmse_rel": 1e-3}


def test_e2b_spearman_is_reported_not_judged():
    e2_rows, rows = _e2b_fixture(treehill_psnr={1e-2: 23.31}, garden_psnr={1e-2: 23.19},
                                 rho0=lambda s, k: {"measured_test_clamped": repr(6e-4)})
    v = e2b.judge_e2b(rows, e2_rows)["per_scene"]["flowers"]["4096"]
    odd = [4e-4, 3e-4, 2e-4, 5e-4]
    # CV test dMSE is 1.1 x the odd-view score: a perfect rank agreement
    assert v["spearman_odd_vs_cv_test"] == pytest.approx(1.0)
    # Amendment 9's version takes E2's gn_vq row (1e-4) at rho = 0; the own-row version E2b's (6e-4)
    assert v["spearman_odd_vs_full_test"] == pytest.approx(gd.spearman(np.array(odd), np.array([1e-4, 3.6e-4, 2.4e-4, 6e-4])))
    assert v["spearman_odd_vs_full_test_own_rho0"] == pytest.approx(
        gd.spearman(np.array(odd), np.array([6e-4, 3.6e-4, 2.4e-4, 6e-4])))


def test_e2b_check_rows_and_the_job():
    """Only E2b's rows reach the criteria (its scenes, configs and rho grid, rho = 0 full-M rows included
    since Amendment 10 c); the job's scenes, grid, columns and pins are Amendment 10's, and a foreign CSV
    is refused."""
    import csv as _csv
    import re

    import gn_e2_scene as e2job
    import gn_e2b_scene as job

    e2_rows, rows = _e2b_fixture()
    assert e2b.check_rows(rows)["per_scene"]["stump"] == {e2b.CV: 8, e2b.FULL: 8}
    for bad in ({**rows[0], "scene": "train"}, {**rows[0], "config": g2.GNVQ}, {**rows[0], "rho": "0.5"}):
        with pytest.raises(RuntimeError, match="not E2b rows"):
            e2b.check_rows(rows + [bad])
    assert e2b.SCENES == ("treehill", "flowers", "stump", "garden") and "train" not in e2b.SCENES
    assert job.K_VALUES == "4096,65536" and job.VQ_EPS == 1e-2 and job.VQ_MAX_ITERS == 20
    assert job.wanted_rows([4096], list(e2b.RHOS)) == [(e2b.CV, 4096, r) for r in e2b.RHOS] + \
        [(e2b.FULL, 4096, r) for r in e2b.RHOS]
    assert len(job.wanted_rows(list(e2b.K_VALUES), list(e2b.RHOS))) == 16
    assert job.COLUMNS[:6] == ["scene", "config", "n_clusters", "seed", "rho", "metric_views"]
    assert set(e2job.COLUMNS) < set(job.COLUMNS) and "measured_odd_clamped" in job.COLUMNS
    repo = os.path.dirname(os.path.dirname(HERE))
    builder = open(os.path.join(repo, "kaggle", "build_gn_e2b_bench.py"), encoding="utf-8").read()
    pinned = dict(re.findall(r'"(\w+)": \("(?:tandt|mipnerf360)", "\w+", "([0-9a-f]{40})"\)', builder))
    run5 = {r["scene"]: r["ckpt_sha1"] for r in _csv.DictReader(
        open(os.path.join(repo, "kaggle", "run5", "tilequant", "run5_results.csv"), newline=""))}
    assert pinned == {s: run5[s] for s in e2b.SCENES} and list(pinned)[:2] == ["treehill", "garden"]
    with pytest.raises(ValueError, match="not an E2b scene"):
        job.main(["--scene", "train", "--dataset", "tandt", "--benchmark_sh", "x", "--data_root", "x",
                  "--ckpt", "x", "--expected_sha1", "x", "--sort_cache_dir", "x", "--warm_dir", "x",
                  "--gn_cache", "x", "--gn_cache_even", "x", "--work_dir", "x", "--runs_dir", "x",
                  "--out_dir", "x", "--examples_dir", "x"])
    path = os.path.join(repo, "kaggle", "gn_e2", "gn2", "gn2_results_treehill.csv")
    with pytest.raises(RuntimeError, match="not an E2b result file"):
        job.assert_e2b_csv(path)  # E2's own CSV is refused


# --------------------------------------------------------------------------- E2c (Amendment 11)

import e2c  # noqa: E402

_E2C_ODD = {0.0: 5e-4, 1e-3: 4e-4, 1e-2: 2e-4, 1e-1: 3e-4, 3e-1: 3.5e-4, 1.0: 6e-4, 3.0: 7e-4}  # rho_cv = 1e-2
_E2C_DMSE = {g2.UPSTREAM: 4e-4, g2.BASELINE: 3e-4, g2.SCALAR: 2e-4, g2.GNVQ: 1e-4}


def _e2c_fixture(final=None, trace=None, wopa=None, odd=None, final_rho=None, drop=(), final_over=None):
    """E2 rows for the five scenes and E2c rows on top of them.

    E2: upstream_l1, lloyd_wopa_area and lloyd_trace on the base curve (``_g2_curve_rows``), gn_vq at 10%
    fewer bytes; ``trace`` / ``wopa`` map a scene to (byte_factor, dpsnr) for those curves. E2c: per K the 7
    CV rows with odd-view scores ``odd`` (rho -> score, default rho 1e-2 lowest; a callable (scene, k) ->
    dict also works), then the final row at their argmin (or ``final_rho``), equal to E2's gn_vq row unless
    ``final[scene]`` = (byte_factor, dpsnr) against the base curve or ``final_over(scene, k)`` overrides
    fields. ``drop`` removes (scene, config, K, rho) rows, rho None for a final row."""
    e2_rows, rows = [], []
    for s in e2c.SCENES:
        for c in g2.CONFIGS:
            bf, dp = {g2.SCALAR: (trace or {}).get(s, (1.0, 0.0)), g2.BASELINE: (wopa or {}).get(s, (1.0, 0.0)),
                      g2.GNVQ: (0.9, 0.0)}.get(c, (1.0, 0.0))
            for r in _g2_curve_rows(s, c, bf, dp):
                e2_rows.append({**r, "PSNR": repr(r["PSNR"]), "size_bytes": str(r["size_bytes"]),
                                "measured_test_clamped": repr(_E2C_DMSE[c]), "SSIM": "0.8", "LPIPS": "0.2"})
        for k in e2c.K_VALUES:
            sc = odd(s, k) if callable(odd) else (odd or _E2C_ODD)
            for rho in e2c.RHOS:
                rows.append({"scene": s, "config": e2c.CV, "n_clusters": str(k), "seed": "0", "rho": repr(rho),
                             "measured_odd_clamped": repr(sc[rho]), "measured_test_clamped": repr(sc[rho] * 1.1),
                             "size_bytes": "15000000"})
            rho_cv = final_rho if final_rho is not None else min(e2c.RHOS, key=lambda r: (sc[r], r))
            gv = next(r for r in e2_rows if r["scene"] == s and r["config"] == g2.GNVQ and r["n_clusters"] == str(k))
            fr = {**gv, "config": e2c.FINAL, "rho": repr(rho_cv), "train_PSNR": "26.0",
                  "m_source": "restored_cache", "warm_start_source": "e2_work_cache"}
            if final and s in final:
                bf, dp = final[s]
                i = list(e2c.K_VALUES).index(k)
                fr.update(PSNR=repr(_G2_BASE_PSNR[i] + dp), size_bytes=str(int(round(_G2_BASE_BYTES[i] * bf))))
            fr.update((final_over or (lambda s_, k_: {}))(s, k))
            rows.append(fr)

    def key(r):
        return (r["scene"], r["config"], int(r["n_clusters"]), None if r["config"] == e2c.FINAL else float(r["rho"]))

    return e2_rows, [r for r in rows if key(r) not in set(drop)]


def test_e2c_grid_labels_and_selection():
    assert e2c.RHOS == (0.0, 1e-3, 1e-2, 1e-1, 3e-1, 1.0, 3.0) and e2c.K_VALUES == g2.K_VALUES
    assert e2c.SCENES == ("bonsai", "counter", "kitchen", "room", "truck")
    assert not set(e2c.SCENES) & (set(g2.DEV) | set(e2b.SCENES) | {"train"})  # no scene a decision used
    assert [e2c.rho_label(r) for r in e2c.RHOS] == ["0", "1e-3", "1e-2", "1e-1", "3e-1", "1", "3"]
    assert e2c.select_rho_cv(_E2C_ODD) == 1e-2
    assert e2c.select_rho_cv({**_E2C_ODD, 1e-1: 2e-4}) == 1e-2  # exact tie: the smaller rho
    assert e2c.select_rho_cv({**_E2C_ODD, 3.0: 1e-5}) == 3.0
    assert e2c.select_rho_cv({r: v for r, v in _E2C_ODD.items() if r != 3.0}) is None
    assert e2c.select_rho_cv({**_E2C_ODD, 1.0: float("nan")}) is None
    assert e2c.on_grid(0.30000000000000004) == 0.3 and e2c.on_grid(0.5) is None


def test_e2c_g2c_every_path():
    """Amendment 11 d: (1) all 5 scenes win against lloyd_trace (G2a's rule), (2) mean BD-rate against
    lloyd_wopa_area <= -5% (Amendment 8's substitutes), (3) BD-PSNR against E2's gn_vq >= -0.01 dB on every
    scene, undefined failing; domain-scaled fit, 9-decimal rounding; incomplete > fail > pass."""
    e2_rows, rows = _e2c_fixture()  # the final curve is E2's gn_vq: 10% fewer bytes than both baselines
    res = e2c.judge_e2c(rows, e2_rows)
    c = res["conditions"]
    assert res["verdict"] == "pass" and res["missing"] == []
    assert c["1_all_win_vs_lloyd_trace"]["n_wins"] == 5 and c["1_all_win_vs_lloyd_trace"]["ok"]
    assert c["2_mean_bd_rate_vs_lloyd_wopa_area"]["mean"] == pytest.approx(-10.0, abs=1e-6)
    assert all(b == pytest.approx(0.0, abs=1e-9) for b in c["3_no_harm_vs_gn_vq"]["bd_psnr"].values())
    v = res["per_scene"]["room"]["vs"][g2.SCALAR]
    assert v["bd_rate"] == g2.bd_rate_scaled(_G2_BASE_BYTES, _G2_BASE_PSNR,
                                             [int(round(b * 0.9)) for b in _G2_BASE_BYTES], _G2_BASE_PSNR)
    assert res["rho_cv"]["truck"] == {str(k): "1e-2" for k in e2c.K_VALUES}
    assert res["n_cells"] == 20 and res["n_cells_rho_cv_above_0"] == 20 and res["n_cells_rho_cv_at_top_of_grid"] == 0
    # (1) fails alone: lloyd_trace 15% cheaper than the base on one scene, so gn_vq_cvfloor loses there
    e2_rows, rows = _e2c_fixture(trace={"counter": (0.85, 0.0)})
    c = e2c.judge_e2c(rows, e2_rows)["conditions"]
    assert not c["1_all_win_vs_lloyd_trace"]["ok"] and c["1_all_win_vs_lloyd_trace"]["per_scene"]["counter"] == "loss"
    assert c["2_mean_bd_rate_vs_lloyd_wopa_area"]["ok"] and c["3_no_harm_vs_gn_vq"]["ok"]
    # (2) fails alone: lloyd_wopa_area only 3% more expensive than the final curve
    e2_rows, rows = _e2c_fixture(wopa={s: (0.93, 0.0) for s in e2c.SCENES})
    res = e2c.judge_e2c(rows, e2_rows)
    c = res["conditions"]
    assert res["verdict"] == "fail" and not c["2_mean_bd_rate_vs_lloyd_wopa_area"]["ok"]
    assert c["1_all_win_vs_lloyd_trace"]["ok"] and c["3_no_harm_vs_gn_vq"]["ok"]
    # (3): exactly 0.01 dB below E2's gn_vq holds, a hair more does not
    for dp, ok in ((-0.01, True), (-0.0100001, False)):
        e2_rows, rows = _e2c_fixture(final={"kitchen": (0.9, dp)})
        res = e2c.judge_e2c(rows, e2_rows)
        c3 = res["conditions"]["3_no_harm_vs_gn_vq"]
        assert c3["per_scene_ok"]["kitchen"] is ok and c3["ok"] is ok and res["verdict"] == ("pass" if ok else "fail")
    # (3): no shared byte range with E2's gn_vq (half the bytes) -> BD-PSNR undefined -> not met
    e2_rows, rows = _e2c_fixture(final={"bonsai": (0.5, 0.0)})
    res = e2c.judge_e2c(rows, e2_rows)
    assert math.isnan(res["conditions"]["3_no_harm_vs_gn_vq"]["bd_psnr"]["bonsai"])
    assert not res["conditions"]["3_no_harm_vs_gn_vq"]["per_scene_ok"]["bonsai"] and res["verdict"] == "fail"
    assert res["conditions"]["1_all_win_vs_lloyd_trace"]["ok"]  # its BD-rate against lloyd_trace is defined
    # (1) with no PSNR overlap: the BD-PSNR decides; (2) takes Amendment 8's substitute
    e2_rows, rows = _e2c_fixture(final={"room": (1.0, 1.0)})
    res = e2c.judge_e2c(rows, e2_rows)
    v = res["per_scene"]["room"]["vs"]
    assert math.isnan(v[g2.SCALAR]["bd_rate"]) and v[g2.SCALAR]["decided_by"] == "bd_psnr" and v[g2.SCALAR]["win"]
    assert v[g2.BASELINE]["mean_term_source"] == "substitute_a"
    assert v[g2.BASELINE]["mean_term"] == pytest.approx(-(1 - _G2_BASE_BYTES[0] / _G2_BASE_BYTES[-1]) * 100)
    assert res["conditions"]["2_mean_bd_rate_vs_lloyd_wopa_area"]["terms"]["room"]["source"] == "substitute_a"


def test_e2c_incomplete_and_the_final_rows_rho():
    # a missing final row, a missing CV row, and a final row at a rho its CV rows do not select
    cases = (
        ([("truck", e2c.FINAL, 16384, None)], None, "truck gn_vq_cvfloor K=16384"),
        ([("room", e2c.CV, 4096, 3.0)], None, "room gn_vq_cvfloor_cv K=4096 rho=3"),
        ((), lambda s, k: {"rho": repr(1e-1)} if (s, k) == ("bonsai", 1024) else {}, "the CV rows select 1e-2"),
    )
    for drop, over, text in cases:
        e2_rows, rows = _e2c_fixture(drop=drop, final_over=over)
        res = e2c.judge_e2c(rows, e2_rows)
        assert res["verdict"] == "incomplete" and any(text in m for m in res["missing"]), res["missing"]
        assert not res["conditions"]["1_all_win_vs_lloyd_trace"]["ok"]
        assert not res["conditions"]["3_no_harm_vs_gn_vq"]["ok"]
    # a missing E2 gate comparator is incomplete; a missing upstream_l1 row (reported only) is not
    e2_rows, rows = _e2c_fixture()
    no_trace = [r for r in e2_rows
                if not (r["scene"] == "kitchen" and r["config"] == g2.SCALAR and r["n_clusters"] == "4096")]
    assert e2c.judge_e2c(rows, no_trace)["verdict"] == "incomplete"
    no_up = [r for r in e2_rows if not (r["scene"] == "kitchen" and r["config"] == g2.UPSTREAM)]
    res = e2c.judge_e2c(rows, no_up)
    assert res["verdict"] == "pass" and g2.UPSTREAM not in res["per_scene"]["kitchen"]["vs"]


def test_e2c_reported_items():
    """Amendment 11 e: rho_cv and the top of the grid, dMSE ratios, LPIPS / SSIM changes, bytes, the CV
    Spearman, the reproduction at rho_cv = 0, the sign and monotonicity flags; none changes the verdict."""
    def top(s, k):
        return {**_E2C_ODD, 3.0: 1e-5} if s == "room" else _E2C_ODD

    e2_rows, rows = _e2c_fixture(odd=top, final_over=lambda s, k: {
        "measured_test_clamped": repr(0.8e-4), "LPIPS": "0.201", "SSIM": "0.799"})
    res = e2c.judge_e2c(rows, e2_rows)
    assert res["verdict"] == "pass" and res["n_cells_rho_cv_at_top_of_grid"] == 4
    cell = res["cells"]["room"]["65536"]
    assert cell["rho_cv"] == 3.0 and cell["rho_cv_at_top_of_grid"] and cell["rho_cv_label"] == "3"
    assert cell["test_dmse_over_gn_vq"] == pytest.approx(0.8) and cell["test_dmse_over_lloyd_trace"] == pytest.approx(0.4)
    assert cell["dLPIPS_vs_gn_vq"] == pytest.approx(0.001) and cell["dSSIM_vs_gn_vq"] == pytest.approx(-0.001)
    assert cell["bytes_ratio"][g2.GNVQ] == 0.0 and cell["bytes_ratio"][g2.SCALAR] == pytest.approx(-0.1, abs=1e-6)
    assert cell["spearman_cv_odd_vs_cv_test"] == pytest.approx(1.0)  # CV test dMSE is 1.1 x the odd score
    assert cell["reproduction_rho0"]["status"] == "not_applicable"
    flags = res["per_scene"]["room"]["vs"][g2.SCALAR]
    assert not flags["sign_disagreement"] and flags["new_rises_with_K"] and flags["ref_rises_with_K"]
    # rho_cv = 0: the final row is E2's gn_vq run again, identical here; a 0.01 dB change is flagged
    zero = {**_E2C_ODD, 0.0: 1e-5}
    e2_rows, rows = _e2c_fixture(odd=zero)
    res = e2c.judge_e2c(rows, e2_rows)
    assert res["n_cells_rho_cv_above_0"] == 0 and res["verdict"] == "pass"
    assert all(v["status"] == "identical" for v in res["reproduction_rho0"]["cells"].values())
    e2_rows, rows = _e2c_fixture(odd=zero, final_over=lambda s, k: {"PSNR": repr(_G2_BASE_PSNR[0] + 0.01)}
                                 if (s, k) == ("truck", 1024) else {})
    rep = e2c.judge_e2c(rows, e2_rows)["reproduction_rho0"]
    assert rep["cells"]["truck/1024"]["status"] == "not_reproduced" and rep["flagged"] == ["truck/1024"]


def test_e2c_check_rows_and_the_job(tmp_path):
    """Only E2c's rows reach G2c; the job's scenes, grid, columns and pins are Amendment 11's; E2's and
    E2b's CSVs are refused; E2's inputs are required before any download."""
    import csv as _csv
    import re

    import gn_e2b_scene as e2bjob
    import gn_e2c_scene as job

    e2_rows, rows = _e2c_fixture()
    assert e2c.check_rows(rows)["per_scene"]["room"] == {e2c.CV: 28, e2c.FINAL: 4}
    for bad in ({**rows[0], "scene": "garden"}, {**rows[0], "config": e2b.CV}, {**rows[0], "rho": "0.5"},
                {**rows[0], "rho": ""}):
        with pytest.raises(RuntimeError, match="not E2c rows"):
            e2c.check_rows(rows + [bad])
    assert job.K_VALUES == "1024,4096,16384,65536" and job.VQ_EPS == 1e-2 and job.VQ_MAX_ITERS == 20
    assert job.wanted_rows([4096], list(e2c.RHOS)) == [(e2c.CV, 4096, r) for r in e2c.RHOS] + [(e2c.FINAL, 4096, None)]
    assert len(job.wanted_rows(list(e2c.K_VALUES), list(e2c.RHOS))) == 32
    assert job.COLUMNS[:len(e2bjob.COLUMNS)] == e2bjob.COLUMNS and job.COLUMNS[-2:] == ["rho_cv", "cv_odd_scores"]
    assert job.row_key({"config": e2c.FINAL, "n_clusters": "4096", "rho": "0.1"}) == (e2c.FINAL, 4096, None)
    repo = os.path.dirname(os.path.dirname(HERE))
    builder = open(os.path.join(repo, "kaggle", "build_gn_e2c_bench.py"), encoding="utf-8").read()
    pinned = dict(re.findall(r'"(\w+)": \("(?:tandt|mipnerf360)", "\w+", "([0-9a-f]{40})"\)', builder))
    run5 = {r["scene"]: r["ckpt_sha1"] for r in _csv.DictReader(
        open(os.path.join(repo, "kaggle", "run5", "tilequant", "run5_results.csv"), newline=""))}
    assert pinned == {s: run5[s] for s in e2c.SCENES} and "ALLOW_WITHOUT" not in builder
    prereg = open(os.path.join(repo, "kaggle", "PREREG_GN.md"), encoding="utf-8").read()
    a11 = prereg[prereg.index("## Amendment 11"):]
    assert dict(re.findall(r"\| (\w+) \| [^|]+ \| `([0-9a-f]{40})` \|", a11)) == pinned
    with pytest.raises(ValueError, match="not an E2c scene"):
        job.main(["--scene", "treehill", "--dataset", "mipnerf360", "--benchmark_sh", "x", "--data_root", "x",
                  "--ckpt", "x", "--expected_sha1", "x", "--sort_cache_dir", "x", "--warm_dir", "x",
                  "--gn_cache", "x", "--gn_cache_even", "x", "--work_dir", "x", "--runs_dir", "x",
                  "--out_dir", "x", "--examples_dir", "x"])
    for path in (os.path.join(repo, "kaggle", "gn_e2", "gn2", "gn2_results_room.csv"),
                 os.path.join(repo, "kaggle", "gn_e2b", "gn2b", "gn2b_results_garden.csv")):
        with pytest.raises(RuntimeError, match="not an E2c result file"):
            job.assert_e2c_csv(path)
    args = argparse.Namespace(gn_cache=str(tmp_path / "none.pt"), warm_dir=str(tmp_path), n_clusters=65536, seed=0)
    with pytest.raises(RuntimeError, match="MISSING E2 INPUT"):
        job.check_inputs_exist(args, [1024, 65536])
    (tmp_path / "e2_kmeans").mkdir()
    (tmp_path / "e2_kmeans" / "lloyd_wopa_area_s0.pt").write_bytes(b"")
    (tmp_path / "none.pt").write_bytes(b"")
    with pytest.raises(RuntimeError, match="K=1024") as exc:
        job.check_inputs_exist(args, [1024, 65536])
    assert "K=65536" not in str(exc.value)  # E2's copied run-4 cache covers K = 65,536


def test_findings_section_11_numbers_recheck_from_the_repo():
    import check_s11

    res = check_s11.run()
    assert res["fails"] == [] and res["n_numbers"] > 100

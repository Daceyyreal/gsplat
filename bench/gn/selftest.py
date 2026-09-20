"""The E0 notebook's smoke tests (kaggle/PREREG_GN.md validity checks), written to gn_selftest.json.
It exits non-zero if a validity check fails, which stops the notebook before the scene jobs.

    python bench/gn/selftest.py --device cuda --out /kaggle/working/gn/gn_selftest.json

First, before anything is rendered, ``fixtures``: the committed scene fixtures of the toy and
end-to-end checks must hash to their pinned values (Amendment 4); otherwise nothing else runs and the
script exits non-zero.

Validity checks (``pass``):
- ``sh_basis``: the SH basis against gsplat's CUDA ``spherical_harmonics`` (max abs error < 1e-5);
- ``toy_exactness``: the toy Hutchinson check (Amendment 1), on its fixture; it also logs the
  report-only ``noise_diagnostic`` (Amendment 4) before it is judged;
- ``e2e_exactness``: predicted = measured unclamped dMSE on the non-overlapping toy (Amendment 3), on
  its fixture, once its preconditions hold on the renders. A failed precondition stops the run as
  "the exactness claim does not apply", not as a mismatch (Amendment 4).

Engineering checks, which also stop the run but are not G0 validity entries:
- ``linalg_scale``: the chunked batched linalg helper (``batched.batched_linalg``) at the sizes and in
  the dtypes the job uses. E0's first Kaggle run died in cuSOLVER's batched eigendecomposition at a
  batch of 65,536 (see ``bench/gn/batched.py``), which no CPU test can catch.

Informational: ``lifted_random``, lifted fp32 vs direct float64 assignment on random data with the
Amendment-3 criterion. The job repeats it on 10,000 real splats, where it gates the refines only.

``--device cpu`` is the CPU stand-in for tests and dry runs: gsplat's pure-torch SH reference and the
brute-force CPU renderer (``toy_render.py``) instead of the CUDA kernels. Without an installed gsplat,
run it with ``PYTHONPATH=<repo>`` (the script never puts the source tree ahead of an installed wheel).
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batched as bl  # noqa: E402
import diagnostics as gd  # noqa: E402
import gn_metric as gm  # noqa: E402
import sh_basis as sb  # noqa: E402

VALIDITY = ("sh_basis", "toy_exactness", "e2e_exactness")  # G0 validity checks
ENGINEERING = ("linalg_scale",)  # must pass, but not part of G0
# What the job runs on: 1,000,000 splats and K = 65,536 clusters.
SCALE_N_SPLATS, SCALE_N_CLUSTERS = 1_000_000, 65536
SCALE_N_SPLATS_CPU, SCALE_N_CLUSTERS_CPU = 8192, 1024


def psd_packed(n: int, device: str, seed: int, chunk: int = 131072) -> torch.Tensor:
    """``[n, 120]`` packed PSD 15x15 matrices (fp32, the job's layout for M), drawn on the CPU in
    chunks; every sixth is the zero matrix and the others are rank-deficient (rank 1..5)."""
    g = torch.Generator().manual_seed(seed)
    out = torch.empty(n, gm.N_PACKED, dtype=torch.float32, device=device)
    ranks = torch.arange(n) % 6
    for start in range(0, n, chunk):
        size = min(chunk, n - start)
        a = torch.randn(size, gm.D, 5, generator=g)
        keep = torch.arange(5)[None, None, :] < ranks[start : start + size, None, None]
        a = (a * keep).to(device)
        out[start : start + size] = gm.pack(a @ a.transpose(1, 2))
    return out


def linalg_scale(device: str) -> dict:
    """The batched linalg the job does, at the job's size: the spectrum's eigendecomposition over
    1,000,000 packed PSD float64 matrices (zero and rank-deficient included) through
    ``diagnostics.eigen_stats``, and the refines' centroid solve over 65,536 systems through
    ``diagnostics.update_centroids``, both variants. Small sizes on CPU."""
    n, k = (
        (SCALE_N_SPLATS, SCALE_N_CLUSTERS)
        if device == "cuda"
        else (SCALE_N_SPLATS_CPU, SCALE_N_CLUSTERS_CPU)
    )
    seen_fallbacks = len(bl.linalg_fallbacks())  # report only this check's own reductions
    t0 = time.perf_counter()
    M = psd_packed(n, device, seed=0)
    stats = gd.eigen_stats(M)  # float64 eigvalsh through the helper
    t_eig = time.perf_counter() - t0
    trace, pr = stats["trace"], stats["pr"]
    pos = trace > 0
    n_zero = int((~pos).sum())
    eig_ok = bool(
        torch.isfinite(trace).all()
        and (trace >= 0).all()
        and n_zero > 0
        and pos.any()
        and ((pr[pos] >= 1 - 1e-6) & (pr[pos] <= gm.D + 1e-6)).all()
        and torch.isnan(pr[~pos]).all()
    )
    del M, stats
    t0 = time.perf_counter()
    Mk = psd_packed(k, device, seed=1)
    g = torch.Generator().manual_seed(2)
    x = (torch.randn(k, gm.D * 3, generator=g) * 0.3).to(device)
    C = (torch.randn(k, gm.D * 3, generator=g) * 0.3).to(device)
    labels = torch.arange(k, device=device)  # one splat per cluster: k systems
    expect_kept = int((gm.trace_packed(Mk) <= 0).sum())
    solve_ok, kept_per_variant = True, {}
    for variant in gd.REFINE_VARIANTS:
        new, kept = gd.update_centroids(x, labels, Mk, C, variant)
        kept_per_variant[variant] = kept
        solve_ok = solve_ok and bool(torch.isfinite(new).all()) and kept == expect_kept
    t_solve = time.perf_counter() - t0
    return {
        "device": device,
        "max_batch": bl.LINALG_MAX_BATCH,
        "eigvalsh": {
            "n": n,
            "dtype": "float64",
            "n_zero_trace": n_zero,
            "time_s": t_eig,
            "ok": eig_ok,
        },
        "solve": {
            "n_systems": k,
            "dtype": "float64",
            "variants": list(gd.REFINE_VARIANTS),
            "clusters_kept": kept_per_variant,
            "clusters_kept_expected": expect_kept,
            "time_s": t_solve,
            "ok": solve_ok,
        },
        "fallbacks": bl.linalg_fallbacks()[seen_fallbacks:],
        "pass": bool(eig_ok and solve_ok),
    }


def check_fixtures() -> dict:
    """Each scene fixture against its pinned hash (read at call time), before any render."""
    out = {}
    for name, pinned in (
        (gm.TOY_SCENE_FIXTURE, gm.TOY_SCENE_SHA256),
        (gd.E2E_SCENE_FIXTURE, gd.E2E_SCENE_SHA256),
    ):
        try:
            _, meta = gm.load_fixture(name, pinned)
            out[name] = {
                "ok": True,
                "sha256": pinned,
                "drawn_under": {
                    k: meta.get(k) for k in ("cpu_capability", "torch_version", "platform")
                },
            }
        except (gm.FixtureHashMismatch, OSError, KeyError, ValueError) as e:
            out[name] = {"ok": False, "sha256": pinned, "error": f"{type(e).__name__}: {e}"}
    return out


def failure_message(out: dict) -> str:
    bad = [k for k, v in out["fixtures"].items() if not v["ok"]]
    if bad:
        return (
            "FIXTURE HASH MISMATCH: "
            + "; ".join(out["fixtures"][k]["error"] for k in bad)
            + ". The checks would not render the pre-registered scenes (PREREG_GN.md Amendment 4); "
            "nothing was rendered and the run stops."
        )
    parts = []
    for k in ENGINEERING:
        r = out.get(k, {"pass": True})
        if not r["pass"]:
            parts.append(
                f"{k}: {r.get('error', 'checks failed')} (batched linalg at the job's size; an "
                "engineering check, not a G0 validity check)"
            )
    for k in VALIDITY:
        r = out[k]
        if r["pass"]:
            continue
        if k == "e2e_exactness" and r.get("status") == "preconditions_failed":
            parts.append(f"e2e_exactness: PRECONDITIONS FAILED, {r['message']}")
        elif k == "e2e_exactness":
            rel = r.get("non_overlapping", {}).get("rel_err")
            parts.append(
                f"e2e_exactness: MISMATCH, predicted vs measured relative error {rel} > "
                f"{r.get('tol_rel')}"
            )
        else:
            parts.append(f"{k} failed")
    return "GN SELFTEST FAILED: " + "; ".join(parts) + ". The run stops."


def lifted_random(device: str) -> dict:
    """Lifted vs direct assignment on random data, M_i of every rank 0..15 (drawn on the CPU)."""
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(4000, 15, 3, generator=g) * 0.3).reshape(4000, 45)
    a = torch.randn(4000, 15, 15, generator=g, dtype=torch.float64)
    a = a * (torch.arange(15)[None, None, :] < (torch.arange(4000) % 16)[:, None, None])
    M = gm.pack(a @ a.transpose(1, 2)).float()
    C = x[torch.randperm(4000, generator=g)[:2048]] + 0.05 * torch.randn(
        2048, 45, generator=g
    )
    return gd.lifted_check(
        x.to(device), M.to(device), C.to(device), n_sample=2000, seed=0
    )


def run(device: str = "cuda") -> dict:
    out = {"device": device, "fixtures": check_fixtures()}
    if not all(v["ok"] for v in out["fixtures"].values()):
        out["pass"] = False  # nothing runs on a scene that is not the pre-registered one
        return out
    if device == "cpu":
        import toy_render as tr

        render = tr.render_bruteforce
        sh = {**sb.reference_check(), "against": "gsplat pure-torch reference (CPU)"}
    else:
        render = None  # gn_metric.gsplat_render
        sh = {**sb.cuda_check(), "against": "gsplat CUDA spherical_harmonics"}
    out["sh_basis"] = sh
    out["toy_exactness"] = gm.toy_exactness(render=render, device=device)
    out["e2e_exactness"] = gd.e2e_exactness(render=render, device=device)
    try:  # an engineering check: record a backend failure instead of losing the whole file
        out["linalg_scale"] = linalg_scale(device)
    except (torch._C._LinAlgError, RuntimeError) as err:
        out["linalg_scale"] = {
            "pass": False,
            "device": device,
            "max_batch": bl.LINALG_MAX_BATCH,
            "fallbacks": bl.linalg_fallbacks(),
            "error": f"{type(err).__name__}: {str(err).splitlines()[0][:300]}",
        }
    out["pass"] = bool(
        all(out[k]["pass"] for k in VALIDITY)
        and all(out[k]["pass"] for k in ENGINEERING)
    )
    out["lifted_random"] = lifted_random(device)  # informational
    return out


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = p.parse_args(argv)
    out = run(args.device)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    if not out["pass"]:
        raise SystemExit(failure_message(out))
    return out


if __name__ == "__main__":
    main()

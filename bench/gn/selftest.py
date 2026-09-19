"""The E0 notebook's smoke tests (kaggle/PREREG_GN.md validity checks), written to gn_selftest.json.
It exits non-zero if a validity check fails, which stops the notebook before the scene jobs.

    python bench/gn/selftest.py --device cuda --out /kaggle/working/gn/gn_selftest.json

Validity checks (``pass``):
- ``sh_basis``: the SH basis against gsplat's CUDA ``spherical_harmonics`` (max abs error < 1e-5);
- ``toy_exactness``: the toy Hutchinson check (Amendment 1);
- ``e2e_exactness``: predicted = measured unclamped dMSE on the non-overlapping toy (Amendment 3).

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

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diagnostics as gd  # noqa: E402
import gn_metric as gm  # noqa: E402
import sh_basis as sb  # noqa: E402

VALIDITY = ("sh_basis", "toy_exactness", "e2e_exactness")


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
    if device == "cpu":
        import toy_render as tr

        render = tr.render_bruteforce
        sh = {**sb.reference_check(), "against": "gsplat pure-torch reference (CPU)"}
    else:
        render = None  # gn_metric.gsplat_render
        sh = {**sb.cuda_check(), "against": "gsplat CUDA spherical_harmonics"}
    out = {"device": device, "sh_basis": sh}
    out["toy_exactness"] = gm.toy_exactness(render=render, device=device)
    out["e2e_exactness"] = gd.e2e_exactness(render=render, device=device)
    out["pass"] = bool(all(out[k]["pass"] for k in VALIDITY))
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
        failed = [k for k in VALIDITY if not out[k]["pass"]]
        raise SystemExit(f"GN SELFTEST FAILED: {failed}")
    return out


if __name__ == "__main__":
    main()

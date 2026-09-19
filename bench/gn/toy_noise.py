"""Probe noise of the toy exactness check (PREREG_GN.md, Amendment 1), simulated on the CPU renderer.

For a fixed toy scene the exact weights ``w_pi`` come from an identity-feature render; each draw of
64 Rademacher probes gives the summed Hutchinson estimate ``sum_i mean_c (sum_p w_pi r_pc)^2``. This
reports how far that sum lands from the exact ``sum_i sum_p w_pi^2`` over many draws, for the toy
layout first written and for the one ``gn_metric.toy_scene`` uses.

    python bench/gn/toy_noise.py  # writes bench/gn/toy_noise.json
"""

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import gn_metric as gm  # noqa: E402
import toy_render as tr  # noqa: E402

SETTINGS = gm.RenderSettings(False, "classic", "pinhole", False, False, 0.01, 1e10, 3)
N_DRAWS = 1000
N_PROBES = 64


def first_layout(seed: int):
    """The toy scene as first written: clustered splats of very different size and opacity."""
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(256, 3, generator=g) * torch.tensor([0.6, 0.45, 0.3])
    means[:, 2] += 3.0
    splats = {
        "means": means,
        "quats": torch.randn(256, 4, generator=g),
        "scales": torch.randn(256, 3, generator=g) * 0.3 - 2.3,
        "opacities": torch.randn(256, generator=g) * 1.5,
    }
    K = torch.tensor([[70.0, 0.0, 32.0], [0.0, 70.0, 24.0], [0.0, 0.0, 1.0]])
    return gm.activated(splats), K, 64, 48


def final_layout(seed: int):
    cam = gm.toy_camera()
    return gm.activated(gm.toy_scene(256, seed)), cam["K"], cam["width"], cam["height"]


def simulate(act, K, width, height, draw_seed: int = 123):
    with torch.no_grad():
        ident, _ = tr.render_bruteforce(
            act, torch.eye(256), torch.eye(4), K, width, height, SETTINGS
        )
    w = ident.reshape(-1, 256).double()
    exact = w.pow(2).sum()
    gen = torch.Generator().manual_seed(draw_seed)
    errs = []
    for _ in range(N_DRAWS):
        r = torch.randint(0, 2, (w.shape[0], N_PROBES), generator=gen).double() * 2 - 1
        errs.append(float(((w.t() @ r).pow(2).mean(1).sum() - exact).abs() / exact))
    e = torch.tensor(errs, dtype=torch.float64)
    covered = (w > 1e-3).sum(1)
    return {
        "image": [width, height],
        "draws": N_DRAWS,
        "probes": N_PROBES,
        "mean_rel_err": float(e.mean()),
        "p99_rel_err": float(e.quantile(0.99)),
        "max_rel_err": float(e.max()),
        "fraction_ge_5pct": float((e >= 0.05).double().mean()),
        "covered_pixels_blending_2plus": float(
            (covered >= 2).double().sum() / (covered >= 1).double().sum()
        ),
        "splats_visible": int((w.sum(0) > 0).sum()),
    }


def main():
    out = {
        "first_layout_seed0": simulate(*first_layout(0)),
        "toy_scene_seed0": simulate(*final_layout(0)),  # the scene the CUDA check uses
        "toy_scene_seed1": simulate(*final_layout(1)),
    }
    path = os.path.join(HERE, "toy_noise.json")
    with open(path, "w", newline="\n") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

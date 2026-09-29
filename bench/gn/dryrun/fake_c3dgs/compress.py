"""A CPU stand-in for C3DGS's ``compress.py`` (E3r's tests and dry run): it imports ``compress_gaussians`` by name,
as C3DGS does, loads the INRIA model directory's 30k ``.ply``, computes a deterministic stand-in for C3DGS's
sensitivity (its own generator: the global random streams are C3DGS's VQ's alone), compresses, optionally
"fine-tunes" the colour and geometry tables (the indices untouched), writes the ``.npz``, ``results.json`` and
``times.json``. It never seeds the global generators, like the real script."""

import argparse
import json
import os
import time

import torch

from compression.vq import CompressionSettings, compress_gaussians
from scene.gaussian_model import GET_FEATURES_CALLS, GaussianModel

p = argparse.ArgumentParser()
p.add_argument("--model_path", required=True)
p.add_argument("--source_path", required=True)
p.add_argument("--data_device", default="cuda")
p.add_argument("--output_vq", required=True)
p.add_argument("--finetune_iterations", type=int, default=5000)
p.add_argument("--color_codebook_size", type=int, default=4096)
p.add_argument("--color_importance_include", type=float, default=0.6e-6)
p.add_argument("--gaussian_codebook_size", type=int, default=16)
p.add_argument("--gaussian_importance_include", type=float, default=0.3e-5)
a = p.parse_args()
if os.environ.get("E3R_FAKE_OOM") == a.data_device:
    raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB (fake)")
t0 = time.time()
g = GaussianModel(3)
g.load_ply(os.path.join(a.model_path, "point_cloud", "iteration_30000", "point_cloud.ply"))
n = g._xyz.shape[0]
gen = torch.Generator().manual_seed(123)
ci = torch.rand(n, generator=gen) * 2e-6
ci[torch.arange(n) % 9 == 0] = 0.0  # never seen: pruned
gi = torch.rand(n, generator=gen) * 6e-6
times = {"sensitivity_calculation": time.time() - t0}
t1 = time.time()
compress_gaussians(g, ci, gi,
                   CompressionSettings(a.color_codebook_size, 0.0, a.color_importance_include, 3, 0.8, 512),
                   CompressionSettings(a.gaussian_codebook_size, None, a.gaussian_importance_include, 3, 0.8, 512),
                   True, prune_threshold=0.0)
times["clustering"] = time.time() - t1
if a.finetune_iterations > 0:
    t2 = time.time()
    with torch.no_grad():
        for _ in range(3):
            g._features_rest += torch.randn_like(g._features_rest) * 1e-3
            g._features_dc += torch.randn_like(g._features_dc) * 1e-3
    times["finetune"] = time.time() - t2
it = 30000 + a.finetune_iterations
out = os.path.join(a.output_vq, "point_cloud", f"iteration_{it}", "point_cloud.npz")
t3 = time.time()
g.save_npz(out, sort_morton=True)
times["encode"] = time.time() - t3
times["total"] = sum(times.values())
json.dump(times, open(os.path.join(a.output_vq, "times.json"), "w"))
size = os.path.getsize(out) / 1024 ** 2
json.dump({f"ours_{it}": {"PSNR": 21.0 + size, "SSIM": 0.8, "LPIPS": 0.2, "size": size}},
          open(os.path.join(a.output_vq, "results.json"), "w"), indent=4)
json.dump({"get_features_calls": GET_FEATURES_CALLS[0]}, open(os.path.join(a.output_vq, "fake_calls.json"), "w"))

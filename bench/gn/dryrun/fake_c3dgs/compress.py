"""A CPU stand-in for C3DGS's ``compress.py`` (E3r's and E4p's tests and dry runs), in C3DGS's own shape: it imports
``compress_gaussians`` and ``finetune`` by name, defines ``calc_importance`` and ``render_and_eval`` at module level,
and ``run_vq`` holds ``gaussians``, ``scene`` and the four parameter objects as locals; the save comes before
C3DGS's evaluation (``render_and_eval``, looked up at call time). Its "sensitivity" is a deterministic stand-in (its
own generator: the global random streams are C3DGS's VQ's and its fine-tuning's alone); its "evaluation" is a number
computed from the model's fake-quantized colours (``get_features``, so the observers update as C3DGS's renders make
them). It never seeds the global generators, like the real script."""

import argparse
import json
import os
import time
from types import SimpleNamespace

import torch

from compression.vq import CompressionSettings, compress_gaussians
from finetune import finetune
from scene import Scene
from scene.gaussian_model import GET_FEATURES_CALLS, GaussianModel


def calc_importance(gaussians, scene, pipeline_params):
    n = gaussians._xyz.shape[0]
    gen = torch.Generator().manual_seed(123)
    ci = torch.rand(n, generator=gen) * 2e-6
    ci[torch.arange(n) % 9 == 0] = 0.0  # never seen: pruned
    gi = torch.rand(n, generator=gen) * 6e-6
    return ci, gi


def render_and_eval(gaussians, scene, model_params, pipeline_params):
    with torch.no_grad():
        f = gaussians.get_features.reshape(-1)
        mse = float(((f - 0.1) ** 2).mean()) + 1e-9
        return {"SSIM": 0.8, "PSNR": -10.0 * torch.log10(torch.tensor(mse)).item(), "LPIPS": 0.2}


def run_vq(model_params, optim_params, pipeline_params, comp_params):
    gaussians = GaussianModel(3)
    scene = Scene(model_params, gaussians, load_iteration=comp_params.load_iteration, shuffle=True)
    times = {}
    t0 = time.time()
    color_importance, gaussian_sensitivity = calc_importance(gaussians, scene, pipeline_params)
    times["sensitivity_calculation"] = time.time() - t0
    t1 = time.time()
    compress_gaussians(gaussians, color_importance, gaussian_sensitivity,
                       CompressionSettings(comp_params.color_codebook_size, 0.0, comp_params.color_importance_include, 3,
                                           0.8, 512),
                       CompressionSettings(comp_params.gaussian_codebook_size, None,
                                           comp_params.gaussian_importance_include, 3, 0.8, 512),
                       True, prune_threshold=0.0)
    times["clustering"] = time.time() - t1
    os.makedirs(comp_params.output_vq, exist_ok=True)
    model_params.model_path = comp_params.output_vq
    iteration = scene.loaded_iter + comp_params.finetune_iterations
    if comp_params.finetune_iterations > 0:
        t2 = time.time()
        finetune(scene, model_params, optim_params, comp_params, pipeline_params, testing_iterations=[-1], debug_from=-1)
        times["finetune"] = time.time() - t2
    out = os.path.join(comp_params.output_vq, "point_cloud", f"iteration_{iteration}", "point_cloud.npz")
    t3 = time.time()
    gaussians.save_npz(out, sort_morton=not comp_params.not_sort_morton)
    times["encode"] = time.time() - t3
    times["total"] = sum(times.values())
    json.dump(times, open(os.path.join(comp_params.output_vq, "times.json"), "w"))
    size = os.path.getsize(out) / 1024 ** 2
    metrics = render_and_eval(gaussians, scene, model_params, pipeline_params)
    metrics["size"] = size
    json.dump({f"ours_{iteration}": metrics}, open(os.path.join(comp_params.output_vq, "results.json"), "w"), indent=4)
    json.dump({"get_features_calls": GET_FEATURES_CALLS[0]}, open(os.path.join(comp_params.output_vq, "fake_calls.json"), "w"))


if __name__ == "__main__":
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
    p.add_argument("--load_iteration", type=int, default=30000)
    p.add_argument("--not_sort_morton", action="store_true")
    a = p.parse_args()
    if os.environ.get("E3R_FAKE_OOM") == a.data_device:
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB (fake)")
    model = SimpleNamespace(model_path=a.model_path, source_path=a.source_path, data_device=a.data_device,
                            white_background=False, sh_degree=3)
    comp = SimpleNamespace(output_vq=a.output_vq, finetune_iterations=a.finetune_iterations,
                           color_codebook_size=a.color_codebook_size, color_importance_include=a.color_importance_include,
                           gaussian_codebook_size=a.gaussian_codebook_size,
                           gaussian_importance_include=a.gaussian_importance_include, load_iteration=a.load_iteration,
                           not_sort_morton=a.not_sort_morton)
    run_vq(model, SimpleNamespace(lambda_dssim=0.2), SimpleNamespace(debug=False), comp)

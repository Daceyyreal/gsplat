"""DRAFT, not run: a GPU-memory probe for INRIA's bicycle, garden and stump checkpoints in the C3DGS host (D4).

    python memory_probe.py --scene bicycle --c3dgs /tmp/c3dgs --model /tmp/probe/bicycle/model \
        --source /tmp/data/bicycle --out /kaggle/working/probe [--data_device cpu] [--ft_iters 200]

What it does, inside C3DGS at 2a234af5 (its own functions, imported from its checkout):
0. installs the chunked ``eigh``/``det`` the project records as a C3DGS deviation (Amendment 13 g);
1. loads the 30k checkpoint and the scene (``Scene``, INRIA's split, ``--eval``) with the images on ``--data_device``;
2. C3DGS's sensitivity pass (``compress.calc_importance``: training views only, with gradients);
3. C3DGS's colour and geometry VQ (``compression.vq.compress_gaussians``) at the default colour threshold (j = 0);
4. OGC's Gram-metric k-means on the colour features of the colour-quantized splats (``bench/gn/ogc_derived.py``'s
   ``gram_kmeans_ours``, OGC's arithmetic, K = 4,096, chunk 25,000, 15 iterations) with a **synthetic** 16 x 16 metric of
   the real size, packed on the host and moved to the GPU as one copy (memory depends on the metric's shape, not values);
5. ``--ft_iters`` iterations of C3DGS's own ``finetune`` (training views only).

**It renders no test view and computes no image metric**, and asserts so: ``Scene.getTestCameras`` and C3DGS's
``render_and_eval`` raise if called, and so do C3DGS's ``psnr`` and the LPIPS entry point. Fine-tuning's training loss
(L1 and D-SSIM on training views) is computed inside C3DGS's loop and never logged or written; ``--ft_iters 0`` skips it.
It logs only: per step the peak allocated and reserved GPU bytes and the host RSS peak, the splat counts (total,
colour-quantized), the image bytes per device and the image size, the checkpoint's sha1. Nothing about any view's
content. Licence: ``ogc_derived.py`` is PolyForm Noncommercial (research use, with its notices); C3DGS is under Inria's
licence; no OGC file is used or written.
"""
import argparse
import hashlib
import json
import os
import resource
import sys
import time


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


class Forbidden(RuntimeError):
    pass


def forbid(obj, name, what):
    def raiser(*a, **k):
        raise Forbidden(f"{what} was called: the probe renders no test view and computes no image metric")

    setattr(obj, name, raiser)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=("bicycle", "garden", "stump"))
    ap.add_argument("--c3dgs", required=True)
    ap.add_argument("--model", required=True)  # INRIA layout: cfg_args, cameras.json, point_cloud/iteration_30000/
    ap.add_argument("--source", required=True)  # the dataset folder (images, sparse)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="/tmp/gsplat")
    ap.add_argument("--data_device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--ft_iters", type=int, default=200)
    a = ap.parse_args()
    sys.path[:0] = [a.c3dgs, os.path.join(a.repo, "bench", "gn"), os.path.join(a.repo, "kaggle")]
    import torch

    import e3q_c3dgs_run

    linalg = e3q_c3dgs_run.ChunkedLinalg().install()  # the recorded deviation (Amendment 13 g): chunked eigh and det

    import compress as c3
    import scene as c3scene
    import utils.image_utils as iu
    from arguments import CompressionParams, ModelParams, OptimizationParams, PipelineParams
    from compression.vq import CompressionSettings, compress_gaussians
    from finetune import finetune
    from scene.gaussian_model import GaussianModel

    # the guards: any test view or image metric raises
    forbid(c3scene.Scene, "getTestCameras", "Scene.getTestCameras")
    forbid(c3, "render_and_eval", "C3DGS's render_and_eval")
    forbid(iu, "psnr", "utils.image_utils.psnr")
    import lpipsPyTorch

    forbid(lpipsPyTorch, "lpips", "lpipsPyTorch.lpips")
    # compress.py binds these at import (compress.py:25, :28): guard its own names too
    forbid(c3, "psnr", "compress.psnr")
    forbid(c3, "lpips", "compress.lpips")

    parser = argparse.ArgumentParser()
    mp, op, pp, cp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser), CompressionParams(parser)
    args = parser.parse_args(["-s", a.source, "-m", a.model, "--eval", "--data_device", a.data_device,
                              "--output_vq", os.path.join(a.out, a.scene, "vq"), "--finetune_iterations", str(a.ft_iters)])
    model_p, opt_p, pipe_p, comp_p = mp.extract(args), op.extract(args), pp.extract(args), cp.extract(args)
    rec = {"scene": a.scene, "data_device": a.data_device, "ft_iters": a.ft_iters, "steps": {},
           "checkpoint_sha1": sha1(os.path.join(a.model, "point_cloud", "iteration_30000", "point_cloud.ply"))}

    def step(name, fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        out = fn()
        torch.cuda.synchronize()
        rec["steps"][name] = {"peak_allocated": torch.cuda.max_memory_allocated(),
                              "peak_reserved": torch.cuda.max_memory_reserved(), "seconds": time.time() - t,
                              "host_rss_peak_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        print(name, rec["steps"][name], flush=True)
        return out

    g = GaussianModel(model_p.sh_degree, quantization=not opt_p.not_quantization_aware)
    sc = step("load", lambda: c3scene.Scene(model_p, g, load_iteration=comp_p.load_iteration, shuffle=True))
    cams = sc.train_cameras[1.0]  # the attribute, not getTestCameras
    img_bytes = {}
    for cam in cams + sc.test_cameras[1.0]:  # loaded by Scene, never rendered here
        im = cam.original_image
        img_bytes[str(im.device)] = img_bytes.get(str(im.device), 0) + im.numel() * im.element_size()
    rec.update(n_splats=int(g.get_xyz.shape[0]), image_bytes=img_bytes,
               image_size=list(cams[0].original_image.shape[1:]), n_train=len(cams), n_test=len(sc.test_cameras[1.0]))
    ci, gs = step("sensitivity", lambda: c3.calc_importance(g, sc, pipe_p))
    colour = CompressionSettings(codebook_size=comp_p.color_codebook_size, importance_prune=comp_p.color_importance_prune,
                                 importance_include=comp_p.color_importance_include,
                                 steps=int(comp_p.color_cluster_iterations), decay=comp_p.color_decay,
                                 batch_size=comp_p.color_batch_size)
    geom = CompressionSettings(codebook_size=comp_p.gaussian_codebook_size, importance_prune=None,
                               importance_include=comp_p.gaussian_importance_include,
                               steps=int(comp_p.gaussian_cluster_iterations), decay=comp_p.gaussian_decay,
                               batch_size=comp_p.gaussian_batch_size)
    keep = ci.amax(-1) > comp_p.color_importance_include
    n_vq = int((~keep).sum())
    rec["n_colour_quantized_before_prune"] = n_vq

    def vq():
        with torch.no_grad():
            compress_gaussians(g, ci.amax(-1), gs.amax(-1), colour, geom, comp_p.color_compress_non_dir,
                               prune_threshold=comp_p.prune_threshold)
    step("c3dgs_vq", vq)

    def ogc():
        import ogc_derived

        torch.manual_seed(0)
        n, d = n_vq, 16
        import gn_metric

        rows = []
        for i in range(0, n, 250_000):  # a synthetic PSD metric of the real size (A A^T + I), packed as E5 packs it
            a_ = torch.randn(min(250_000, n - i), d, 4) / 2.0
            rows.append(gn_metric.pack((a_ @ a_.transpose(1, 2) + torch.eye(d)).float()))
        m_rows = torch.cat(rows)  # host memory, as E5's call receives it
        rec["metric_host_bytes"] = m_rows.numel() * m_rows.element_size()
        x48 = torch.randn(n, d * 3)
        return ogc_derived.gram_kmeans_ours(x48, m_rows, 4096, device="cuda", chunk=25_000, iters=15)
    step("ogc_vq_synthetic_metric", ogc)
    if a.ft_iters > 0:
        step("finetune", lambda: finetune(sc, model_p, opt_p, comp_p, pipe_p, testing_iterations=[-1], debug_from=-1))
    rec["chunked_linalg_calls"] = dict(linalg.n_calls)
    rec["peak_allocated_max"] = max(s["peak_allocated"] for s in rec["steps"].values())
    rec["peak_reserved_max"] = max(s["peak_reserved"] for s in rec["steps"].values())
    rec["t4_total"] = torch.cuda.get_device_properties(0).total_memory
    os.makedirs(a.out, exist_ok=True)
    json.dump(rec, open(os.path.join(a.out, f"probe_{a.scene}_{a.data_device}.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in rec.items() if k != "steps"}, indent=2))


if __name__ == "__main__":
    main()

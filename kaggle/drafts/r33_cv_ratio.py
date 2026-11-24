"""DRAFT (R33), not run: a training-views-only transfer ratio for OGC's Gram VQ against C3DGS's own, per scene.

    python r33_cv_ratio.py --scene bonsai --c3dgs /tmp/c3dgs --model /tmp/model/bonsai --source /tmp/data/bonsai \
        --out /kaggle/working/r33 [--repo /tmp/gsplat] [--k 4] [--seed 0]

**Definition (kaggle/drafts/R33_DECLARATION_DRAFT.md).** The scene's training views (INRIA's split: C3DGS's ``Scene``
with ``--eval``, the cameras sorted by image name) are split into ``k`` = 4 folds by a seeded permutation
(``numpy.random.default_rng(seed).permutation``, view at permuted position ``p`` in fold ``p % k``). For each fold f:

1. **in-fold** = the other k - 1 folds' views, **held-out** = fold f's views; no test view is read at any point;
2. C3DGS's sensitivity pass on the in-fold views only, then C3DGS's colour and geometry VQ at E5's middle rate point
   (j = 0, C3DGS's default colour threshold, K = 4,096), with E3r's recording hooks (``kaggle/e3r_hooks.py``), which
   save the colour VQ's input ``x``, the colour-quantized ids, C3DGS's codebook and labels and its int8 quantizer state;
3. the 16 x 16 GN metric (``bench/gn/gn_metric.compute_gn(..., with_dc=True)``, this project's 16-probe estimate, probe
   seed 0) over the in-fold views only, on the uncompressed model;
4. OGC's Gram VQ by this project's derived code (``bench/gn/ogc_derived.gram_kmeans_ours``: OGC's arithmetic, metric
   "gram", K = 4,096, 15 iterations, lam 1e-3, chunk 25,000, seed 0) on the same ``x`` with that metric;
5. both codebooks through C3DGS's int8 table quantizer (``bench/gn/e3r.C3DGSQuantizer``), substituted into the
   uncompressed model's colours for the colour-quantized splats (the kept splats keep C3DGS's kept rows), geometry
   unchanged, as E3r's and E4q's training-view scoring does;
6. each row rendered by C3DGS's own rasterizer (``gaussian_renderer.render``) on every in-fold and held-out view,
   quantized to 8 bits, PSNR against the view's image as C3DGS's ``Scene`` loads it (the dataset's reduced images at
   ``cfg_args``' resolution, as protocol ii);
7. ``gain_in(f)`` and ``gain_out(f)``: the mean over the in-fold and held-out views of PSNR(ogc_gram) - PSNR(c3dgs).

The scene's ratio is ``mean_f gain_out(f) / gain_in(f)``; the ratio of the fold means is reported beside it.

**It reads no test view and computes no test-view metric,** and asserts so (``install_guards``): the scene readers drop
the test camera infos before any image is decoded (``drop_test_infos``), ``Scene.getTestCameras``
and C3DGS's ``render_and_eval`` raise if called, so do C3DGS's ``psnr`` and the LPIPS entry point (both modules' bindings),
and the loaded ``Scene``'s ``test_cameras`` is replaced by a mapping that raises on any access. Licence: ``ogc_derived.py``
is PolyForm Noncommercial (research use, with its notices); C3DGS is under Inria's licence; no OGC file is used.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time

K_FOLDS = 4
SEED = 0
K_CODEBOOK = 4096
LAM = 1e-3
CHUNK = 25_000
ITERS = 15


class Forbidden(RuntimeError):
    pass


def forbid(obj, name, what):
    def raiser(*a, **k):
        raise Forbidden(f"{what} was called: R33 reads no test view and computes no test-view metric")

    setattr(obj, name, raiser)


class RaisingCameras(dict):
    """Stands in for ``Scene.test_cameras``: any access raises (R33's guard)."""

    def _no(self, *a, **k):
        raise Forbidden("Scene.test_cameras was accessed: R33 reads no test view")

    __getitem__ = __iter__ = __len__ = __contains__ = get = keys = values = items = _no


DROPPED = []  # the number of test camera infos dropped per loaded scene (a count only)


def drop_test_infos(callbacks):
    """Wrap C3DGS's scene readers (``sceneLoadTypeCallbacks``, a dict ``Scene`` reads by key) so the test camera infos
    are dropped before ``Scene`` turns them into cameras: no test image's pixels are ever decoded. (C3DGS's reader opens
    every image lazily to read its size before it splits the views; that reads headers, not pixels.)"""
    for key, fn in list(callbacks.items()):
        def wrapped(*a, _fn=fn, **k):
            info = _fn(*a, **k)
            DROPPED.append(len(info.test_cameras))
            return info._replace(test_cameras=[])

        callbacks[key] = wrapped


def install_guards(c3, c3scene, iu, lpips_mod):
    """The test-view guards, before any scene is loaded."""
    drop_test_infos(c3scene.sceneLoadTypeCallbacks)
    forbid(c3scene.Scene, "getTestCameras", "Scene.getTestCameras")
    forbid(c3, "render_and_eval", "C3DGS's render_and_eval")
    forbid(iu, "psnr", "utils.image_utils.psnr")
    forbid(lpips_mod, "lpips", "lpipsPyTorch.lpips")
    forbid(c3, "psnr", "compress.psnr")  # compress.py binds these at import (compress.py:25, :28)
    forbid(c3, "lpips", "compress.lpips")


def guard_scene(sc):
    """After ``Scene`` is built: its test cameras are unreachable."""
    sc.test_cameras = RaisingCameras()
    return sc


def folds(n: int, k: int = K_FOLDS, seed: int = SEED):
    """Fold of each of ``n`` views (sorted by image name): position p of a seeded permutation goes to fold p % k."""
    import numpy as np

    perm = np.random.default_rng(seed).permutation(n)
    f = [0] * n
    for p, v in enumerate(perm):
        f[int(v)] = p % k
    return f


def psnr8(render, gt) -> float:
    """PSNR of the render quantized to 8 bits against the image ([3, H, W] in [0, 1]), as protocol ii."""
    import torch

    r = (render.clamp(0.0, 1.0) * 255.0).round() / 255.0
    mse = torch.mean((r - gt.clamp(0.0, 1.0)) ** 2).item()
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def ratio(gains):
    """``gains``: per fold (gain_in, gain_out). Returns the mean of the per-fold ratios, the ratio of the means, and
    the folds whose in-fold gain is not positive (their ratio is undefined and they are left out of the mean)."""
    per = [(go / gi) if gi > 0 else None for gi, go in gains]
    ok = [r for r in per if r is not None]
    mi = sum(g[0] for g in gains) / len(gains)
    mo = sum(g[1] for g in gains) / len(gains)
    return {"per_fold": per, "mean_of_ratios": (sum(ok) / len(ok)) if ok else None,
            "ratio_of_means": (mo / mi) if mi > 0 else None,
            "undefined_folds": [i for i, r in enumerate(per) if r is None]}


def camera_view(cam):
    """A C3DGS camera as ``gn_metric.compute_gn``'s view dict (camtoworld, K, width, height)."""
    import torch

    w2c = cam.world_view_transform.detach().transpose(0, 1).double()  # C3DGS stores it transposed
    c2w = torch.linalg.inv(w2c).float()
    W, H = int(cam.image_width), int(cam.image_height)
    fx = W / (2.0 * math.tan(float(cam.FoVx) / 2.0))
    fy = H / (2.0 * math.tan(float(cam.FoVy) / 2.0))
    K = torch.tensor([[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]])
    return {"camtoworld": c2w.cpu(), "K": K, "width": W, "height": H}


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--c3dgs", required=True)
    ap.add_argument("--model", required=True)  # INRIA layout: cfg_args, cameras.json, point_cloud/iteration_30000/
    ap.add_argument("--source", required=True)
    ap.add_argument("--images", required=True)  # cfg_args' images folder (images_2, images_4 or images)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="/tmp/gsplat")
    ap.add_argument("--data_device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--k", type=int, default=K_FOLDS)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    sys.path[:0] = [a.c3dgs, os.path.join(a.repo, "bench", "gn"), os.path.join(a.repo, "kaggle")]
    import torch

    import e3p_inria as ei
    import e3q_c3dgs_run

    linalg = e3q_c3dgs_run.ChunkedLinalg().install()  # the recorded deviation (Amendment 13 g)
    import compress as c3
    import lpipsPyTorch
    import scene as c3scene
    import utils.image_utils as iu

    install_guards(c3, c3scene, iu, lpipsPyTorch)
    import compression.vq as cvq
    import e3r
    import e3r_hooks
    import gn_metric as gm
    import ogc_derived
    from arguments import CompressionParams, ModelParams, OptimizationParams, PipelineParams
    from compression.vq import CompressionSettings
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel

    parser = argparse.ArgumentParser()
    mp, op, pp, cp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser), CompressionParams(parser)
    args = parser.parse_args(["-s", a.source, "-m", a.model, "-i", a.images, "--eval", "--data_device", a.data_device])
    model_p, opt_p, pipe_p, comp_p = mp.extract(args), op.extract(args), pp.extract(args), cp.extract(args)
    out_dir = os.path.join(a.out, a.scene)
    os.makedirs(out_dir, exist_ok=True)
    rec = {"scene": a.scene, "k": a.k, "seed": a.seed, "K": K_CODEBOOK, "lam": LAM, "iters": ITERS, "chunk": CHUNK,
           "checkpoint_sha1": sha1(os.path.join(a.model, "point_cloud", "iteration_30000", "point_cloud.ply")),  # noqa
           "images": a.images, "data_device": a.data_device, "folds": [], "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    g0 = GaussianModel(model_p.sh_degree, quantization=not opt_p.not_quantization_aware)
    sc = guard_scene(c3scene.Scene(model_p, g0, load_iteration=comp_p.load_iteration, shuffle=False))
    cams = sorted(sc.train_cameras[1.0], key=lambda c: c.image_name)
    fold_of = folds(len(cams), a.k, a.seed)
    rec.update(n_test_dropped=DROPPED[-1], n_train=len(cams), fold_sizes=[fold_of.count(i) for i in range(a.k)],
               fold_of={c.image_name: f for c, f in zip(cams, fold_of)})
    settings = gm.RenderSettings(packed=False, rasterize_mode="classic", camera_model="pinhole", with_ut=False,
                                 with_eval3d=False, near_plane=0.01, far_plane=1e10, sh_degree=3)  # E5's runner
    bg = torch.zeros(3, device="cuda")
    colour = CompressionSettings(codebook_size=comp_p.color_codebook_size, importance_prune=comp_p.color_importance_prune,
                                 importance_include=comp_p.color_importance_include,
                                 steps=int(comp_p.color_cluster_iterations), decay=comp_p.color_decay,
                                 batch_size=comp_p.color_batch_size)
    geom = CompressionSettings(codebook_size=comp_p.gaussian_codebook_size, importance_prune=None,
                               importance_include=comp_p.gaussian_importance_include,
                               steps=int(comp_p.gaussian_cluster_iterations), decay=comp_p.gaussian_decay,
                               batch_size=comp_p.gaussian_batch_size)
    assert colour.codebook_size == K_CODEBOOK, colour.codebook_size
    ply = os.path.join(a.model, "point_cloud", "iteration_30000", "point_cloud.ply")
    uncompressed = ei.read_inria_ply(ply)[0]  # gsplat's splat dict, in the ply's order (the harness's loader, as E5)

    for f in range(a.k):
        t0 = time.time()
        in_cams = [c for c, i in zip(cams, fold_of) if i != f]
        out_cams = [c for c, i in zip(cams, fold_of) if i == f]
        g = copy.deepcopy(g0)  # the uncompressed model, fresh for each fold
        sc.gaussians = g
        full_train = sc.train_cameras[1.0]
        sc.train_cameras[1.0] = in_cams  # C3DGS's sensitivity pass sees the in-fold views only
        try:
            ci, gs = c3.calc_importance(g, sc, pipe_p)
        finally:
            sc.train_cameras[1.0] = full_train
        record = os.path.join(out_dir, f"probe_fold{f}.pt")
        hooks = e3r_hooks.install(record=record)
        torch.manual_seed(a.seed)
        with torch.no_grad():
            cvq.compress_gaussians(g, ci.amax(-1), gs.amax(-1), colour, geom, comp_p.color_compress_non_dir,
                                   prune_threshold=comp_p.prune_threshold)
        for name, fn in hooks.orig.items():  # uninstall: the next fold starts from C3DGS's own functions
            if name == "save_npz":
                GaussianModel.save_npz = fn
            else:
                setattr(cvq, name, fn)
        pr = torch.load(record, map_location="cpu", weights_only=False)
        ids = pr["vq_ids"].long()
        with torch.enable_grad():
            gn = gm.compute_gn({k: v.cuda() for k, v in uncompressed.items()}, [camera_view(c) for c in in_cams],
                               settings, seed=0, log=None, with_dc=True)
        M_rows = gn["M_packed"].cpu().index_select(0, ids)
        del gn
        x = pr["x"].float()
        C, L, info = ogc_derived.gram_kmeans_ours(x, M_rows, K_CODEBOOK, LAM, "cuda", chunk=CHUNK, metric="gram")
        q = e3r.C3DGSQuantizer.from_state(pr["qa"])
        base = e3r.colours_of(uncompressed["sh0"], uncompressed["shN"]).cpu().clone()
        base[pr["kept_ids"].long()] = pr["kept_rows"].to(base.dtype)
        rows = {"c3dgs": (q.quantize(pr["codebook"].float())[0], pr["labels"].long()),
                "ogc_gram": (q.quantize(C.float().cpu())[0], L.long().cpu())}
        psnr = {}
        # the evaluation model: not quantization-aware (its fake quantizers would re-quantize the substituted colours)
        g_eval = GaussianModel(model_p.sh_degree, quantization=False)
        g_eval.load_ply(ply)
        for row, (Cq, Lr) in rows.items():
            v = base.clone()
            v[ids] = Cq[Lr].to(v.dtype)
            sh0, shn = e3r.split_colours(v)
            g_eval._features_dc = torch.nn.Parameter(sh0.cuda().contiguous())
            g_eval._features_rest = torch.nn.Parameter(shn.cuda().contiguous())
            with torch.no_grad():
                psnr[row] = {c.image_name: psnr8(render(c, g_eval, pipe_p, bg)["render"], c.original_image.cuda())
                             for c in in_cams + out_cams}
        gi = sum(psnr["ogc_gram"][c.image_name] - psnr["c3dgs"][c.image_name] for c in in_cams) / len(in_cams)
        go = sum(psnr["ogc_gram"][c.image_name] - psnr["c3dgs"][c.image_name] for c in out_cams) / len(out_cams)
        rec["folds"].append({"fold": f, "n_in": len(in_cams), "n_out": len(out_cams), "gain_in": gi, "gain_out": go,
                             "n_colour_quantized": int(ids.numel()), "ogc": info, "psnr": psnr,
                             "seconds": time.time() - t0})
        json.dump(rec, open(os.path.join(out_dir, f"r33_{a.scene}.json"), "w"), indent=1, default=str)
        os.remove(record)
        del g, g_eval, M_rows, x, C, L
        torch.cuda.empty_cache()
        print(f"{a.scene} fold {f}: gain_in {gi:+.4f}, gain_out {go:+.4f} ({time.time() - t0:.0f} s)", flush=True)
    rec["ratio"] = ratio([(r["gain_in"], r["gain_out"]) for r in rec["folds"]])
    rec["chunked_linalg_calls"] = dict(linalg.n_calls)
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(rec, open(os.path.join(out_dir, f"r33_{a.scene}.json"), "w"), indent=1, default=str)
    print(json.dumps({"scene": a.scene, "ratio": rec["ratio"]}, indent=1))


if __name__ == "__main__":
    main()

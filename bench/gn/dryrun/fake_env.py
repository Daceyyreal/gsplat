"""The CPU stand-in the E0 and E1 dry runs share: a toy checkpoint, its sort and clustering caches,
and a fake ``simple_trainer.Runner``.

Nothing here is GPU code. ``gm.gsplat_render`` is replaced by ``bench/gn/toy_render.py``, TorchPQ
(not installed locally) by the library's Lloyd, and ``tilequant_sweep.build_runner`` by ``FakeRunner``
(toy scene, brute-force CPU renderer, an eval that writes stats, and metric modules). Everything the
dry runs actually test - ``PngCompression``, the library ``weighted_kmeans``, the GN metric, the
diagnostics, GN-VQ, the refines, G0 and G1 - is the real code.

    import fake_env as fe
    env = fe.build(root)          # toy checkpoint, sort cache, run-3 caches
    fe.patch_all()                # render, TorchPQ stand-in, build_runner
"""

import hashlib
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for path in (REPO, os.path.join(REPO, "kaggle"), os.path.join(REPO, "bench", "gn")):
    if path not in sys.path:
        sys.path.insert(0, path)
import gn_metric as gm  # noqa: E402
import tilequant_run3 as r3  # noqa: E402
import tilequant_sweep as ts  # noqa: E402
import toy_render as tr  # noqa: E402
from gsplat.compression.kmeans import weighted_kmeans  # noqa: E402

N, H, W = 64 * 64, 24, 32  # 4,096 splats: 64 x 64, so PngCompression needs no crop
KS, KDEF = [16, 32, 64], 64  # the dry runs' K grid; KDEF stands in for the run-3 default
KMAT = torch.tensor([[30.0, 0, W / 2], [0, 30.0, H / 2], [0, 0, 1]])
FAKE = {}  # the last FakeRunner built, for assertions


class FakeCfg:
    packed = False
    antialiased = False
    camera_model = "pinhole"
    with_ut = False
    with_eval3d = False
    near_plane = 0.01
    far_plane = 1e10
    sh_degree = 3
    app_opt = False
    post_processing = None
    sh_fp16 = False


class FakeDataset:
    def __init__(self, cams, gt):
        self.cams, self.gt = cams, gt

    def __len__(self):
        return len(self.cams)

    def __getitem__(self, i):
        return {
            "camtoworld": self.cams[i],
            "K": KMAT,
            "image": self.gt[i] * 255.0,
            "camera_idx": 0,
            "image_id": i,
        }


def cam(tx, ty):
    c = torch.eye(4)
    c[0, 3], c[1, 3] = tx, ty
    return c


def _psnr(a, b):
    return -10 * torch.log10(((a - b) ** 2).mean())


class FakeRunner:
    def __init__(self, work_dir, splats):
        self.cfg = FakeCfg()
        self.device = torch.device("cpu")
        self.splats = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v.clone()) for k, v in splats.items()}
        )
        self.stats_dir = os.path.join(work_dir, "runner", "stats")
        os.makedirs(self.stats_dir, exist_ok=True)
        self.psnr = _psnr
        self.ssim = lambda a, b: torch.tensor(0.5) + 0 * a.mean()
        self.lpips = lambda a, b: (a - b).abs().mean()
        train = [cam(0.1 * i, -0.05 * i) for i in range(5)]
        test = [cam(-0.07, 0.03), cam(0.2, 0.1)]
        with torch.no_grad():
            gt = [
                self.rasterize_splats(c[None], KMAT[None], W, H, sh_degree=3)[0][0].clamp(0, 1)
                + 0.02
                for c in train + test
            ]
        self.trainset = FakeDataset(train, gt[:5])
        self.valset = FakeDataset(test, gt[5:])
        self.n_eval = 0

    def rasterize_splats(
        self,
        camtoworlds,
        Ks,
        width,
        height,
        sh_degree=3,
        near_plane=0.01,
        far_plane=1e10,
        masks=None,
        frame_idcs=None,
        camera_idcs=None,
        exposure=None,
        splats=None,
        **kw,
    ):
        s = splats if splats is not None else self.splats
        act = gm.activated(s)
        coeffs = torch.cat([s["sh0"], s["shN"]], 1).detach()
        settings = gm.RenderSettings.from_cfg(self.cfg)
        img, info = tr.render_bruteforce(
            act, coeffs, camtoworlds[0], Ks[0], width, height, settings, sh_degree=sh_degree
        )
        return img[None], None, info

    def eval(self, step, stage):
        # as Runner.eval: per-image metrics on the val views, then the mean
        vals = {"psnr": [], "ssim": [], "lpips": []}
        for i in range(len(self.valset)):
            d = self.valset[i]
            with torch.no_grad():
                img = self.rasterize_splats(
                    d["camtoworld"][None], d["K"][None], W, H, sh_degree=3
                )[0].clamp(0, 1)
            c, p = img.permute(0, 3, 1, 2), (d["image"] / 255.0)[None].permute(0, 3, 1, 2)
            vals["psnr"].append(self.psnr(c, p))
            vals["ssim"].append(self.ssim(c, p))
            vals["lpips"].append(self.lpips(c, p))
        self.n_eval += 1
        stats = {k: torch.stack(v).mean().item() for k, v in vals.items()}
        stats["num_GS"] = N
        json.dump(stats, open(os.path.join(self.stats_dir, f"{stage}_step{step:04d}.json"), "w"))


def fake_torchpq(data, n_clusters, seed, distance, max_iter):
    """TorchPQ is not installed locally; the library's Lloyd stands in for recomputed L1 codebooks."""
    c, l = weighted_kmeans(data, n_clusters, max_iter=5, seed=seed)
    return c, l, [{}] * 5


def fake_build_runner(args):
    r = FakeRunner(args.work_dir, FAKE["splats"])
    FAKE["runner"] = r
    return r, 29999


def patch_all():
    """Point the GN render, the TorchPQ clustering and the runner at the CPU stand-ins."""
    gm.gsplat_render = tr.render_bruteforce
    r3.torchpq_kmeans = fake_torchpq
    ts.build_runner = fake_build_runner


def toy_splats(seed: int = 0):
    """The toy checkpoint's splats."""
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(N, 2, generator=g) * 2 - 1
    return {
        "means": torch.stack(
            [xy[:, 0] * 1.5, xy[:, 1] * 1.1, 3.0 + torch.randn(N, generator=g) * 0.3], -1
        ),
        "quats": torch.randn(N, 4, generator=g),
        "scales": torch.randn(N, 3, generator=g) * 0.2 - 3.2,
        "opacities": torch.randn(N, generator=g),
        "sh0": torch.randn(N, 1, 3, generator=g) * 0.5,
        "shN": torch.randn(N, 15, 3, generator=g) * 0.2,
    }


def build(root: str, seeds=(0,), stale_lloyd_w1: bool = True) -> dict:
    """Write the toy checkpoint, its seed-0 sort cache and the run-3 clustering caches under ``root``.

    ``manhattan_log`` and ``lloyd_wopa_area`` are cached for every seed in ``seeds``; with
    ``stale_lloyd_w1`` a ``lloyd_w1`` cache with a wrong key is written too, so a dry run sees one
    config fall back to reclustering."""
    splats = toy_splats()
    FAKE["splats"] = splats
    ckpt = os.path.join(root, "ckpt.pt")
    torch.save({"splats": splats, "step": 29999}, ckpt)
    sha = ts.file_sha1(ckpt)
    g = torch.Generator().manual_seed(1)
    order = torch.randperm(N, generator=g)
    sort_dir = os.path.join(root, "sort")
    os.makedirs(os.path.join(sort_dir, "seed0"), exist_ok=True)
    json.dump({"key": sha, "seeds": {"0": {}}}, open(os.path.join(sort_dir, "cache_info.json"), "w"))
    torch.save(order, os.path.join(sort_dir, "seed0", "order.pt"))
    key3 = hashlib.sha1(sha.encode() + order.numpy().tobytes()).hexdigest()

    sorted_raw = {k: v[order] for k, v in splats.items()}
    x = sorted_raw["shN"].reshape(N, -1)
    km_dir = os.path.join(root, "run3", "kmeans")
    os.makedirs(km_dir, exist_ok=True)
    weights = r3.cluster_weights("opacity_area", sorted_raw)
    for seed in seeds:
        c_l1, l_l1 = weighted_kmeans(x, KDEF, max_iter=5, seed=seed)
        torch.save(
            {"key": key3, "centroids": c_l1, "labels": l_l1, "time_s": 1.0, "log": [{}] * 5},
            os.path.join(km_dir, f"manhattan_log_s{seed}.pt"),
        )
        c_w, l_w = weighted_kmeans(x, KDEF, weights=weights, max_iter=20, seed=seed)
        torch.save(
            {"key": key3, "centroids": c_w, "labels": l_w, "time_s": 2.0, "log": [{}] * 20},
            os.path.join(km_dir, f"lloyd_wopa_area_s{seed}.pt"),
        )
        if stale_lloyd_w1 and seed == 0:
            torch.save(
                {"key": "stale", "centroids": c_w, "labels": l_w},
                os.path.join(km_dir, "lloyd_w1_s0.pt"),
            )
    return {
        "splats": splats,
        "ckpt": ckpt,
        "sha": sha,
        "order": order,
        "sort_dir": sort_dir,
        "key3": key3,
        "km_dir": km_dir,
        "sorted_raw": sorted_raw,
    }

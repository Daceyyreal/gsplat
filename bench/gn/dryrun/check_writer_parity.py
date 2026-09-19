"""E0's writer (the library's builtin path fed a precomputed codebook, ``gn_e0_scene.write_and_decode``)
against the run-3 writer (``ShnBenchCompression`` scalar mode, i.e. the TorchPQ path with a
precomputed stub): the same files, byte for byte, for the same codebook. CPU only; TorchPQ is
replaced by a stand-in module because only its import is needed.

    python bench/gn/dryrun/check_writer_parity.py

Scratch files go to a fresh system temp directory that is deleted at the end.
"""

import os
import shutil
import sys
import tempfile
import types

import torch

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "kaggle"))
sys.path.insert(0, os.path.join(REPO, "bench", "gn"))
tq = types.ModuleType("torchpq")
tq.clustering = types.ModuleType("torchpq.clustering")
tq.clustering.KMeans = None
sys.modules["torchpq"], sys.modules["torchpq.clustering"] = tq, tq.clustering

import gn_e0_scene as job  # noqa: E402
import tilequant_run3 as r3  # noqa: E402
import tilequant_shn as tsn  # noqa: E402
from gsplat.compression.kmeans import weighted_kmeans  # noqa: E402


def read_dir(path):
    return {
        n: open(os.path.join(path, n), "rb").read() for n in sorted(os.listdir(path))
    }


def main():
    N, K = 64 * 64, 64
    g = torch.Generator().manual_seed(5)
    splats = {
        "means": torch.randn(N, 3, generator=g),
        "quats": torch.randn(N, 4, generator=g),
        "scales": torch.randn(N, 3, generator=g) - 3,
        "opacities": torch.randn(N, generator=g),
        "sh0": torch.randn(N, 1, 3, generator=g),
        "shN": torch.randn(N, 15, 3, generator=g) * 0.2,
    }
    x = splats["shN"].reshape(N, -1)
    root = tempfile.mkdtemp(prefix="gn_writer_parity_")
    try:
        for name, w in (
            ("unweighted", None),
            ("opacity_area", r3.cluster_weights("opacity_area", splats)),
        ):
            c, labels = weighted_kmeans(x, K, weights=w, max_iter=10, seed=0)
            e0_dir = os.path.join(root, name, "e0")
            job.write_and_decode(e0_dir, splats, c, labels)
            run3_dir = os.path.join(root, name, "run3")
            os.makedirs(run3_dir)
            # compress_shn(scalar) calls the library _compress_kmeans with n_clusters = K
            # through the precomputed TorchPQ stub
            tsn.ShnBenchCompression(
                use_sort=False,
                verbose=False,
                shn_mode="scalar",
                codebook={"centroids": c, "labels": labels, "kept_coeffs": 15},
            ).compress(run3_dir, {k: v.clone() for k, v in splats.items()})
            files_e0, files_r3 = read_dir(e0_dir), read_dir(run3_dir)
            diff = sorted(
                n
                for n in set(files_e0) | set(files_r3)
                if files_e0.get(n) != files_r3.get(n)
            )
            print(
                f"{name}: {len(files_e0)} files, identical: {not diff}, differing: {diff}"
            )
            assert not diff, diff
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("WRITER PARITY OK")


if __name__ == "__main__":
    main()

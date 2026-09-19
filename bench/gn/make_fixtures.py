"""Write the committed scene fixtures of the toy and end-to-end checks (kaggle/PREREG_GN.md Amendment 4).

    python bench/gn/make_fixtures.py --verify-commit c69ba388          # writes only missing files
    python bench/gn/make_fixtures.py --verify-commit c69ba388 --force  # overwrites

- ``fixtures/toy_scene_seed0``: ``gn_metric.toy_scene(256, seed=0)``, drawn on the CPU. With
  ``--verify-commit``, the same scene is re-derived with ``gn_metric.py`` from that commit (the commit
  of ``toy_noise.py`` / ``toy_noise.json``) in a separate process; if the hashes differ, nothing is
  written and the script exits non-zero.
- ``fixtures/e2e_scene_seed0``: ``diagnostics.e2e_scene(seed=0)`` and ``diagnostics.e2e_delta(seed=0)``.

Each fixture is an .npz (float32, little-endian) and a .json with its hash and the CPU capability,
torch version and platform of the draw. A fresh draw on another machine can differ in its last bits
(torch's CPU randn depends on the dispatched CPU capability), which is why the checks render these
files. The pinned hashes are ``gn_metric.TOY_SCENE_SHA256`` and ``diagnostics.E2E_SCENE_SHA256``.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import diagnostics as gd  # noqa: E402
import gn_metric as gm  # noqa: E402

REDERIVE = (
    "import sys; sys.path.insert(0, sys.argv[1]); import numpy as np; import gn_metric as gm; "
    "s = gm.toy_scene(256, 0); np.savez(sys.argv[2], **{k: v.numpy() for k, v in s.items()})"
)


def rederive_toy_scene(commit: str) -> dict:
    """``toy_scene(256, 0)`` on the CPU, drawn by ``bench/gn/gn_metric.py`` of ``commit``."""
    tmp = tempfile.mkdtemp(prefix="gn_rederive_")
    for name in ("gn_metric.py", "sh_basis.py"):
        src = subprocess.check_output(
            ["git", "-C", REPO, "show", f"{commit}:bench/gn/{name}"]
        )
        with open(os.path.join(tmp, name), "wb") as f:
            f.write(src)
    out = os.path.join(tmp, "scene.npz")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    subprocess.run(
        [sys.executable, "-c", REDERIVE, tmp, out], check=True, env=env, cwd=tmp
    )
    with np.load(out) as z:
        return {k: torch.from_numpy(z[k].astype(np.float32)) for k in z.files}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--verify-commit", default="")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    toy = gm.toy_scene(256, 0, "cpu")
    toy_meta = {
        "drawn_by": "gn_metric.toy_scene(256, seed=0, device='cpu'), CPU generator",
        "seed": 0,
        "n_splats": 256,
        "used_by": "gn_metric.toy_exactness (toy check, PREREG_GN.md Amendments 1 and 4)",
        "simulated_by": "bench/gn/toy_noise.py -> bench/gn/toy_noise.json (commit c69ba388)",
    }
    if args.verify_commit:
        old = rederive_toy_scene(args.verify_commit)
        old_hash, new_hash = gm.tensors_sha256(old), gm.tensors_sha256(toy)
        print(f"toy scene re-derived at {args.verify_commit}: {old_hash}")
        print(f"toy scene, current code:        {new_hash}")
        if old_hash != new_hash:
            raise SystemExit(
                "SCENES DIFFER: the current toy_scene draw is not the scene of commit "
                f"{args.verify_commit}; nothing written"
            )
        toy_meta.update(
            rederived_from_commit=args.verify_commit, rederived_sha256=old_hash
        )
    e2e_meta = {
        "drawn_by": (
            "diagnostics.e2e_scene(False, seed=0, device='cpu') and "
            "diagnostics.e2e_delta(seed=0), CPU generator"
        ),
        "seed": 0,
        "n_splats": 48,
        "amplitude": gd.E2E_AMPLITUDE,
        "used_by": "diagnostics.e2e_exactness (end-to-end check, PREREG_GN.md Amendments 3 and 4)",
    }
    for name, tensors, meta, pinned in (
        (gm.TOY_SCENE_FIXTURE, toy, toy_meta, gm.TOY_SCENE_SHA256),
        (gd.E2E_SCENE_FIXTURE, gd.e2e_fixture_tensors(0), e2e_meta, gd.E2E_SCENE_SHA256),
    ):
        got = gm.tensors_sha256(tensors)
        if os.path.exists(gm.fixture_paths(name)[0]) and not args.force:
            print(f"{name}: exists, not overwritten (draw here: {got})")
            continue
        full = gm.save_fixture(name, tensors, meta)
        note = "the pinned hash" if got == pinned else f"NOT the pinned {pinned}"
        print(
            f"{name}: wrote {full['sha256']} ({note}); {full['cpu_capability']}, "
            f"torch {full['torch_version']}, {full['platform']}"
        )


if __name__ == "__main__":
    main()

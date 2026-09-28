"""Run C3DGS's ``compress.py`` unchanged, in its own directory, and record this process's wall time and peak GPU
memory from torch's allocator (E3q, kaggle/PREREG_GN.md Amendment 13 c).

    python kaggle/e3q_c3dgs_run.py --c3dgs_dir /tmp/c3dgs --out_json run.json -- <compress.py arguments>

It executes ``compress.py`` with ``runpy`` as ``__main__``, exactly as ``python compress.py <arguments>`` would,
and writes ``{"status", "error", "wall_s", "max_memory_allocated", "max_memory_reserved", "device"}``. It exits 1
if ``compress.py`` raised.
"""

import argparse
import json
import os
import runpy
import sys
import time
import traceback


def main() -> int:
    argv = sys.argv[1:]
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--c3dgs_dir", required=True)
    p.add_argument("--out_json", required=True)
    a = p.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    import torch

    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    os.chdir(a.c3dgs_dir)
    sys.path.insert(0, a.c3dgs_dir)
    sys.argv = ["compress.py"] + rest
    rec = {"argv": sys.argv, "status": "ok", "error": None}
    t0 = time.time()
    try:
        runpy.run_path("compress.py", run_name="__main__")
    except BaseException as e:  # noqa: B902 (recorded, then the exit code says so)
        rec.update(status="error", error=f"{type(e).__name__}: {str(e)[:800]}", traceback=traceback.format_exc()[-6000:])
    rec["wall_s"] = time.time() - t0
    if cuda:
        rec.update(max_memory_allocated=torch.cuda.max_memory_allocated(),
                   max_memory_reserved=torch.cuda.max_memory_reserved(), device=torch.cuda.get_device_name(0))
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    with open(a.out_json, "w") as f:
        json.dump(rec, f, indent=2)
    return 0 if rec["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())

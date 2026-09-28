"""Run C3DGS's ``compress.py`` unchanged, in its own directory, and record this process's wall time and peak GPU
memory from torch's allocator (E3q, kaggle/PREREG_GN.md Amendment 13 c).

    python kaggle/e3q_c3dgs_run.py --c3dgs_dir /tmp/c3dgs --out_json run.json -- <compress.py arguments>

It executes ``compress.py`` with ``runpy`` as ``__main__``, exactly as ``python compress.py <arguments>`` would,
and writes ``{"status", "error", "wall_s", "max_memory_allocated", "max_memory_reserved", "device", "linalg_patch"}``.
It exits 1 if ``compress.py`` raised.

Before that it installs ``ChunkedLinalg`` (Amendment 13 g). E3q's attempt 1 died in C3DGS's
``utils/splats.py:extract_rot_scale``: ``torch.linalg.eigh`` on one batch of 3x3 float32 matrices (the Gaussian
codebook plus every kept splat, up to 1,030,604 on train) was refused by cuSOLVER
(``cusolverDnXsyevBatched_bufferSize`` -> ``CUSOLVER_STATUS_INVALID_VALUE``), E0's failure class. So in this process
``torch.linalg.eigh`` and ``torch.Tensor.det`` (the same function's ``R.det()``, on the same batch) go through
``bench/gn/batched.py``'s ``batched_linalg``: at most 8,192 matrices per call, halved on a backend refusal, every
reduction recorded. C3DGS's source is not edited. Each matrix is decomposed on its own, so chunking changes nothing
that is computed; it is still recorded as a deviation. Each chunked call also gets a report-only check against a
float64 CPU reference on a sample of its actual input matrices.
"""

import argparse
import json
import os
import runpy
import sys
import time
import traceback
from typing import Callable, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_BATCH = 8192  # Amendment 13 g: batched.py's measured eigendecomposition batch (OP_MAX_BATCH)
CHECK_SAMPLE = 4096
CHECK_SEED = 0
MAX_CALL_RECORDS = 100
DEVIATION = (
    "torch.linalg.eigh and torch.Tensor.det were replaced in the wrapper's process, before compress.py ran, by "
    "chunked versions (bench/gn/batched.py's batched_linalg: at most 8,192 matrices per call, halved on a backend "
    "refusal, every reduction recorded; PREREG_GN.md Amendment 13 g); C3DGS's source was not edited; each matrix "
    "is decomposed independently, so what is computed does not change"
)


def _import_batched():
    """``bench/gn/batched.py``, imported without leaving ``bench/gn`` on ``sys.path`` (C3DGS imports by bare names)."""
    saved = list(sys.path)
    sys.path.insert(0, os.path.join(REPO, "bench", "gn"))
    try:
        import batched
    finally:
        sys.path[:] = saved
    return batched


def _like_layout(out, part):
    """``out`` (chunks joined by ``torch.cat``, so row-major) in the memory layout the op itself returns: torch's
    eigh gives column-major eigenvectors, and a later op on them (C3DGS's ``R.det()``) rounds differently on the
    transpose. Values are unchanged; only the strides follow ``part``, the op's own result for the first chunk."""
    import torch

    if isinstance(out, torch.Tensor):
        if out.dim() >= 2 and not part.is_contiguous() and part.mT.is_contiguous():
            return out.mT.contiguous().mT
        return out
    fields = [_like_layout(o, p) for o, p in zip(out, part)]
    try:  # torch.return_types.* are struct sequences: rebuild the same type
        return type(out)(fields)
    except TypeError:
        return tuple(fields)


class ChunkedLinalg:
    """``torch.linalg.eigh`` and ``torch.Tensor.det`` in chunks of the batch dimension, with a record of every call.

    Only a single batch of square matrices (``[N, n, n]``) is chunked; anything else, and a call with ``out=``,
    goes to the original function unchanged. Each chunked call starts at ``max_batch`` (or at the smaller batch that
    already worked for that op in this process) and ``batched_linalg`` halves it on a backend refusal. After each
    chunked call, a report-only check compares up to ``check_sample`` of its matrices, drawn from a separate
    generator (seed ``check_seed``) so the global random stream is untouched, with a float64 CPU reference from the
    original function. A failing check is recorded, never raised."""

    def __init__(self, max_batch: int = MAX_BATCH, check_sample: int = CHECK_SAMPLE, check_seed: int = CHECK_SEED,
                 log: Optional[Callable[[str], None]] = print):
        self.bl = _import_batched()
        self.max_batch, self.check_sample, self.check_seed, self.log = max_batch, check_sample, check_seed, log
        self.orig: Dict[str, Callable] = {}
        self.calls: List[Dict] = []
        self.n_calls = {"eigh": 0, "det": 0}
        self.passthrough = {"eigh": 0, "det": 0}
        self.fallbacks_before = len(self.bl.linalg_fallbacks())
        self.installed = False

    def install(self) -> "ChunkedLinalg":
        import torch

        self.orig = {"eigh": torch.linalg.eigh, "det": torch.Tensor.det}

        def det(A, *args, **kwargs):  # a plain function, so that ``R.det()`` binds R as it does for the method
            return self.det(A, *args, **kwargs)

        torch.linalg.eigh = self.eigh
        torch.Tensor.det = det
        self.installed = True
        return self

    def uninstall(self) -> None:
        import torch

        if self.installed:
            torch.linalg.eigh = self.orig["eigh"]
            torch.Tensor.det = self.orig["det"]
            self.installed = False

    def _start_batch(self, name: str) -> int:
        return min(self.max_batch, self.bl.linalg_working_batches().get(name, self.max_batch))

    def _chunked(self, key: str, x, **kwargs):
        orig = self.orig[key]
        name = getattr(orig, "__name__", key)
        start = self._start_batch(name)
        n_fb = len(self.bl.linalg_fallbacks())
        first: List = []

        def op(t, **kw):  # the original op, remembering the first chunk's result for its memory layout
            r = orig(t, **kw)
            if not first:
                first.append(r)
            return r

        op.__name__ = name  # batched.py records and remembers batches by the op's name
        t0 = time.time()
        out = self.bl.batched_linalg(op, x, max_batch=start, log=self.log, **kwargs)
        out = _like_layout(out, first[0]) if first and out is not first[0] else out
        rec = {"op": name, "n": int(x.shape[0]), "shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device),
               "start_batch": start, "reductions": len(self.bl.linalg_fallbacks()) - n_fb, "time_s": time.time() - t0}
        rec["check"] = self._check(key, x, out, kwargs)
        self.n_calls[key] += 1
        if len(self.calls) < MAX_CALL_RECORDS:
            self.calls.append(rec)
        return out

    def eigh(self, A, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], str) and "UPLO" not in kwargs:  # UPLO given by position
            args, kwargs = (), {**kwargs, "UPLO": args[0]}
        if args or "out" in kwargs or A.dim() != 3 or A.shape[-1] != A.shape[-2] or A.shape[0] == 0:
            self.passthrough["eigh"] += 1
            return self.orig["eigh"](A, *args, **kwargs)
        return self._chunked("eigh", A, **kwargs)

    def det(self, A, *args, **kwargs):
        if args or kwargs or A.dim() != 3 or A.shape[-1] != A.shape[-2] or A.shape[0] == 0:
            self.passthrough["det"] += 1
            return self.orig["det"](A, *args, **kwargs)
        return self._chunked("det", A)

    def _check(self, key: str, x, out, kwargs) -> Dict:
        """The report-only float64 CPU reference on a sample of the call's own input matrices."""
        import torch

        try:
            n = x.shape[0]
            g = torch.Generator().manual_seed(self.check_seed)
            idx = torch.randperm(n, generator=g)[: min(self.check_sample, n)].sort().values
            a = x.detach()[idx.to(x.device)].to(device="cpu", dtype=torch.float64)
            res = {"n_sampled": int(idx.numel()), "seed": self.check_seed, "reference": "float64, CPU, the original op"}
            if key == "eigh":
                uplo = kwargs.get("UPLO", "L")
                ref = self.orig["eigh"](a, UPLO=uplo).eigenvalues
                w = out.eigenvalues.detach()[idx.to(x.device)].to(device="cpu", dtype=torch.float64)
                v = out.eigenvectors.detach()[idx.to(x.device)].to(device="cpu", dtype=torch.float64)
                # the symmetric matrix eigh reads: the UPLO triangle mirrored
                tri = a.triu() if uplo.upper() == "U" else a.tril()
                sym = tri + tri.transpose(-2, -1) - torch.diag_embed(torch.diagonal(a, dim1=-2, dim2=-1))
                recon = v @ torch.diag_embed(w) @ v.transpose(-2, -1)
                res.update(max_abs_eigenvalue_diff=float((w - ref).abs().max()),
                           max_abs_eigenvalue=float(ref.abs().max()),
                           max_abs_reconstruction_residual=float((recon - sym).abs().max()))
            else:
                ref = self.orig["det"](a)
                got = out.detach()[idx.to(x.device)].to(device="cpu", dtype=torch.float64)
                res.update(max_abs_det_diff=float((got - ref).abs().max()), max_abs_det=float(ref.abs().max()))
            res["nonfinite_in_sample"] = int((~torch.isfinite(a)).any(dim=(-2, -1)).sum())
            return res
        except Exception as e:  # report-only: recorded, never raised
            return {"error": f"{type(e).__name__}: {str(e)[:400]}"}

    def record(self) -> Dict:
        return {
            "installed": self.installed, "ops": ["torch.linalg.eigh", "torch.Tensor.det"], "max_batch": self.max_batch,
            "check_sample": self.check_sample, "check_seed": self.check_seed, "n_chunked_calls": dict(self.n_calls),
            "n_passthrough_calls": dict(self.passthrough), "calls": list(self.calls),
            "fallbacks": self.bl.linalg_fallbacks()[self.fallbacks_before:],
            "working_batches": self.bl.linalg_working_batches(), "deviation": DEVIATION,
        }


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
    patch = ChunkedLinalg().install()
    os.chdir(a.c3dgs_dir)
    sys.path.insert(0, a.c3dgs_dir)
    sys.argv = ["compress.py"] + rest
    rec = {"argv": sys.argv, "status": "ok", "error": None, "deviations": [DEVIATION]}
    t0 = time.time()
    try:
        runpy.run_path("compress.py", run_name="__main__")
    except BaseException as e:  # noqa: B902 (recorded, then the exit code says so)
        rec.update(status="error", error=f"{type(e).__name__}: {str(e)[:800]}", traceback=traceback.format_exc()[-6000:])
    rec["wall_s"] = time.time() - t0
    rec["linalg_patch"] = patch.record()
    if cuda:
        rec.update(max_memory_allocated=torch.cuda.max_memory_allocated(),
                   max_memory_reserved=torch.cuda.max_memory_reserved(), device=torch.cuda.get_device_name(0))
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    with open(a.out_json, "w") as f:
        json.dump(rec, f, indent=2)
    return 0 if rec["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())

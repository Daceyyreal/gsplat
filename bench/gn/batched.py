"""Chunked batched ``torch.linalg`` calls: the one place that knows the backend's batch limits.

E0's first Kaggle run (2026-09-20) died in both scene jobs right after the GN pass, in
``diagnostics.eigen_stats``:

    torch._C._LinAlgError: cusolver error: CUSOLVER_STATUS_INVALID_VALUE, when calling
    cusolverDnXsyevBatched_bufferSize(... CUDA_R_64F ..., batchSize)

``eigen_stats`` already looped in chunks of 65,536, so that is the batch cuSOLVER refused, and the
query that failed only validates parameters (it does not read the matrix values). 65,536 is one more
than 65,535, the CUDA limit on a grid dimension, so the batch itself is the suspect; the refines'
centroid solve runs at the same size (one system per cluster, K = 65,536).

So every batched ``torch.linalg`` call in ``bench/gn/`` and ``kaggle/gn_e0_scene.py`` goes through
``batched_linalg``, which keeps each call at ``LINALG_MAX_BATCH`` or below and halves it further if a
backend still refuses (the real limit cannot be checked without a GPU). ``finite_report`` is the guard
to run before any linalg: non-finite input makes these backends fail in ways that look like batch
problems.
"""

import math
import os
import sys
from typing import Callable, Dict, List, Optional

import torch
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Half of the batch cuSOLVER refused, and below the 65,535 grid limit. Configurable per call.
LINALG_MAX_BATCH = 32768
# A backend refusing a batch, or running out of memory, says so in the message; anything else (a
# matrix that does not converge, say) is a real failure and is raised.
_RETRY_MARKERS = (
    "cusolver",
    "cublas",
    "cusparse",
    "magma",
    "invalid value",
    "invalid_value",
    "invalid argument",
    "workspace",
    "buffer",
    "out of memory",
)
_FALLBACKS: List[Dict] = []


def linalg_fallbacks() -> List[Dict]:
    """Every batch reduction ``batched_linalg`` had to make; empty when the default was enough."""
    return list(_FALLBACKS)


def _retryable(err: BaseException) -> bool:
    return any(m in str(err).lower() for m in _RETRY_MARKERS)


def _concat(parts: List):
    """Concatenate chunk results along the batch dimension, keeping the op's return type."""
    first = parts[0]
    if len(parts) == 1:
        return first
    if isinstance(first, Tensor):
        return torch.cat(parts, dim=0)
    fields = [_concat([p[i] for p in parts]) for i in range(len(first))]
    try:  # torch.return_types.* are struct sequences: rebuild the same type
        return type(first)(fields)
    except TypeError:
        return tuple(fields)


def batched_linalg(
    op: Callable,
    *tensors: Tensor,
    max_batch: Optional[int] = None,
    log: Optional[Callable[[str], None]] = print,
    **kwargs,
):
    """``op(*tensors, **kwargs)`` in chunks of the batch dimension, results concatenated.

    Every tensor is sliced along dim 0 and must have the same size there; ``kwargs`` are passed to
    every chunk. A chunk that fails with a batch or memory error from the backend (``_RETRY_MARKERS``)
    is retried at half the batch, down to a single matrix, and the reduction is recorded in
    ``linalg_fallbacks()``. Any other error is raised unchanged."""
    if not tensors:
        raise ValueError("batched_linalg needs at least one batched tensor")
    n = tensors[0].shape[0]
    if any(t.shape[0] != n for t in tensors):
        raise ValueError(
            f"batched_linalg: batch sizes differ: {[tuple(t.shape) for t in tensors]}"
        )
    if n == 0:
        return op(*tensors, **kwargs)
    step = max(1, min(max_batch or LINALG_MAX_BATCH, n))
    parts: List = []
    start = 0
    while start < n:
        size = min(step, n - start)
        try:
            parts.append(op(*(t[start : start + size] for t in tensors), **kwargs))
        except (torch._C._LinAlgError, RuntimeError) as err:
            if size <= 1 or not _retryable(err):
                raise
            name = getattr(op, "__name__", str(op))
            step = max(1, size // 2)
            first_line = str(err).splitlines()[0][:200]
            _FALLBACKS.append(
                {"op": name, "batch": size, "retry_batch": step, "error": first_line}
            )
            if log is not None:
                log(
                    f"batched_linalg: {name} failed at batch {size} ({first_line}); "
                    f"retrying at {step}"
                )
            if "out of memory" in str(err).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        start += size
    return _concat(parts)


def finite_report(x: Tensor, name: str = "M") -> Dict:
    """Whether every entry of a batch is finite, with counts and the first offending rows.

    Run this before any linalg on a tensor built by the GN pass: cuSOLVER and MAGMA report non-finite
    input as opaque backend errors, and non-finite values would be a separate bug anyway."""
    bad = ~torch.isfinite(x)
    per_row = bad.reshape(x.shape[0], -1).any(dim=1) if x.dim() > 1 else bad
    return {
        "name": name,
        "n_rows": int(x.shape[0]),
        "n_nonfinite_rows": int(per_row.sum()),
        "n_nonfinite_entries": int(bad.sum()),
        "n_nan_entries": int(torch.isnan(x).sum()),
        "n_inf_entries": int(torch.isinf(x).sum()),
        "first_nonfinite_rows": [
            int(i) for i in per_row.nonzero(as_tuple=True)[0][:10]
        ],
        "max_abs": float(x.abs().amax()) if x.numel() and bool(torch.isfinite(x).any()) else math.nan,
        "finite": not bool(bad.any()),
    }

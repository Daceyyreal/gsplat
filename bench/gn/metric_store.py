"""One GPU copy of the GN metric for ``gn_vq_cvfloor`` at INRIA scale (kaggle/PREREG_GN.md Amendment 12 a).
A memory layout only: the method's arithmetic is unchanged.

E2c's job holds five ``[N, 120]`` float32 metrics at once: E2's full ``M`` and ``M_even`` as loaded, their
sorted copies ``M[order]`` and ``M_even[order]``, and the floored metric ``e2b.floored_metric``
materializes for the row in progress. At bicycle's 6.13M INRIA splats that is 14.7 GB, beyond a 16 GB T4
(kaggle/E3_SCOUTING.md c). Here:

- the unfloored metrics stay in host memory, unsorted, as the GN pass (or its cache) produced them;
- the GPU holds one ``[n_sorted, 120]`` buffer: the floored metric in use, in the sorted order.
  ``MetricStore.floored(kind, rho)`` refills it slice by slice. Each slice's host rows are gathered with
  the sort order, moved to the device and floored with ``e2b.floored_metric`` itself, whose arithmetic is
  per row (a float64 trace of that row's diagonal, then a float64 add and a cast), so the buffer holds the
  values E2c's ``e2b.floored_metric(M[order], rho)`` holds;
- the unfloored metric is read in two more places: GN-VQ's reported objective under the unfloored ``M``
  (``gn_vq(report_metrics=...)``) and ``P`` (``diagnostics.predicted_dmse``). Both go through
  ``diagnostics.quad_form``, which reads its metric one slice of rows at a time. ``HostMetric`` answers
  those reads from host memory, on the device the slice is asked for, so quad_form sees the slices it
  would have cut from a device tensor.

``bench/gn/test_gn.py`` checks on the CPU that GN-VQ's codebook, labels and report, ``P`` and the lifted
check come out bit-identical to E2c's code path, for every ``rho`` of the grid, both metrics, and slices
that do not divide the problem evenly.

The buffer is refilled in place: a tensor ``floored()`` returned earlier holds the new values after the
next call with another ``(kind, rho)``. The E3p job uses one metric at a time.
"""

import os
import sys
from typing import Dict, Optional

import torch
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2b  # noqa: E402

SLICE = 262144  # rows per host-to-device slice; quad_form's own default chunk


class HostMetric:
    """A packed metric ``[n, 120]`` in host memory, optionally read through ``index`` (the sort order), in
    the one form ``diagnostics.quad_form`` reads a metric: ``shape``, ``device``, and a slice with step 1,
    which returns those rows on ``device``. Anything else raises, so a caller that needs the whole tensor
    on the device fails loudly instead of copying it."""

    def __init__(self, host: Tensor, device, index: Optional[Tensor] = None):
        if host.device.type != "cpu":
            raise ValueError(f"HostMetric holds a host tensor, got one on {host.device}")
        self.host = host
        self.index = None if index is None else index.detach().to("cpu", torch.int64)
        self.device = torch.device(device)

    @property
    def shape(self) -> torch.Size:
        n = self.host.shape[0] if self.index is None else self.index.shape[0]
        return torch.Size((n, self.host.shape[1]))

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key) -> Tensor:
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError(f"HostMetric supports row slices with step 1 only, got {key!r}")
        start, stop, _ = key.indices(len(self))
        rows = self.host[start:stop] if self.index is None else self.host.index_select(0, self.index[start:stop])
        return rows.to(self.device)


class ChunkedDifference:
    """``(a - b).view(-1, *shape)`` built one row slice at a time, in the one form ``diagnostics.quad_form``
    reads its ``delta``: a slice with step 1. E3p's rows pass it to ``predicted_dmse`` instead of the full
    ``[N, 15, 3]`` float32 difference (1.1 GB at 6.13M splats). Each slice holds exactly the values the same
    slice of ``a - b`` holds (the subtraction is element-wise), so ``P`` is unchanged bit for bit."""

    def __init__(self, a: Tensor, b: Tensor, shape=(15, 3)):
        if a.shape != b.shape:
            raise ValueError(f"shapes differ: {tuple(a.shape)} and {tuple(b.shape)}")
        self.a, self.b, self.shape = a, b, tuple(shape)

    def __len__(self) -> int:
        return self.a.shape[0]

    def __getitem__(self, key) -> Tensor:
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError(f"ChunkedDifference supports row slices with step 1 only, got {key!r}")
        return (self.a[key] - self.b[key]).view(-1, *self.shape)


class MetricStore:
    """The unfloored metrics in host memory and the one device buffer (module docstring)."""

    def __init__(self, device, order: Tensor, slice_rows: int = SLICE):
        self.device = torch.device(device)
        self.order = order.detach().to("cpu", torch.int64)
        self.slice_rows = int(slice_rows)
        self.host: Dict[str, Tensor] = {}
        self.buf: Optional[Tensor] = None
        self.holds = None  # the (kind, rho) the buffer holds

    def add(self, kind: str, M: Tensor) -> None:
        """Keep ``M`` (``[N, 120]``, unsorted, as the GN pass wrote it) in host memory; a host tensor is kept
        as it is, a device tensor is copied to the host."""
        self.host[kind] = M if M.device.type == "cpu" else M.detach().cpu()
        if self.holds is not None and self.holds[0] == kind:
            self.holds = None

    def floored(self, kind: str, rho: float) -> Tensor:
        """The device buffer, filled with ``e2b.floored_metric(M_kind[order], rho)``, slice by slice."""
        if self.holds == (kind, rho):
            return self.buf
        host = self.host[kind]
        n = self.order.shape[0]
        if self.buf is None or self.buf.shape != (n, host.shape[1]) or self.buf.dtype != host.dtype:
            self.buf = None
            self.buf = torch.empty(n, host.shape[1], dtype=host.dtype, device=self.device)
        self.holds = None
        for start in range(0, n, self.slice_rows):
            rows = host.index_select(0, self.order[start : start + self.slice_rows]).to(self.device)
            self.buf[start : start + rows.shape[0]] = e2b.floored_metric(rows, rho)
        self.holds = (kind, rho)
        return self.buf

    def sorted(self, kind: str) -> HostMetric:
        """The unfloored metric in the sorted order (for ``gn_vq``'s ``report_metrics``)."""
        return HostMetric(self.host[kind], self.device, self.order)

    def unsorted(self, kind: str) -> HostMetric:
        """The unfloored metric in the splats' own order (for ``predicted_dmse``)."""
        return HostMetric(self.host[kind], self.device)

    def release(self) -> None:
        """Drop the device buffer."""
        self.buf, self.holds = None, None

    def device_bytes(self) -> int:
        return 0 if self.buf is None else self.buf.numel() * self.buf.element_size()

    def host_bytes(self) -> Dict[str, int]:
        return {k: v.numel() * v.element_size() for k, v in self.host.items()}

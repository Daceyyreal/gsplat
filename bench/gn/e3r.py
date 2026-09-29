"""E3r's method pieces (kaggle/PREREG_GN.md Amendment 14 b): GN-VQ with the floored 16 x 16 metric inside C3DGS.

- ``C3DGSQuantizer``: C3DGS's per-tensor int8 quantization of its colour table, DC (the first 3 of the 48 values,
  index ``k * 3 + channel``) and AC (the other 45) with their own scale and zero point, as its two fake quantizers
  hold them. ``quantize`` is ``torch.quantize_per_tensor`` then ``dequantize``, the call C3DGS's ``save_npz`` makes;
  it is ``gn_vq``'s ``quantizer`` here, so the clip and the final exact assignment see the codec's grid.
- ``run_gn_vq``: E2's GN-VQ (eps 1e-2, at most 20 iterations, the 1e-3 relative-drop rule, the clip to the warm
  start's range with per-cluster acceptance) on ``e2b.floored_metric`` of a packed 16 x 16 metric, which floors by
  ``tr(M_i) / 16`` at that width. The harness's cross-validation and the injected C3DGS run both call it.
- ``colour_dmse``: the render-vs-render dMSE when the colours (DC and AC) of some splats change; ``diagnostics.
  measure_dmse`` with ``sh0`` swapped too.
- ``trace_shares``: the colour-quantized splats' share of the total ``tr(M_i)``, under the 16 x 16 metric and its
  15 x 15 block.
"""

import os
import sys
import time
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import diagnostics as gd  # noqa: E402
import e2b  # noqa: E402
import gn_metric as gm  # noqa: E402
import gn_vq as vq  # noqa: E402

N_COLOUR = 48  # 16 coefficients x 3 channels, C3DGS's colour vector with color_compress_non_dir = True
N_DC = 3
VQ_EPS = 1e-2  # Amendment 7 c, as E2 / E2c / E3p ran it
VQ_MAX_ITERS = 20
VQ_REL_TOL = vq.REL_TOL


class C3DGSQuantizer:
    """C3DGS's int8 per-tensor affine quantization of a ``[K, 48]`` colour codebook: columns 0-2 (DC) with
    ``features_dc_qa``'s scale and zero point, columns 3-47 with ``features_rest_qa``'s."""

    def __init__(self, dc_scale: float, dc_zero_point: int, rest_scale: float, rest_zero_point: int):
        self.dc = (float(dc_scale), int(dc_zero_point))
        self.rest = (float(rest_scale), int(rest_zero_point))

    @classmethod
    def from_state(cls, qa: Dict) -> "C3DGSQuantizer":
        return cls(qa["dc_scale"], qa["dc_zero_point"], qa["rest_scale"], qa["rest_zero_point"])

    def state(self) -> Dict:
        return {"dc_scale": self.dc[0], "dc_zero_point": self.dc[1], "rest_scale": self.rest[0],
                "rest_zero_point": self.rest[1]}

    def range(self, C: Tensor) -> Dict:
        """The grid's representable range per part, and the codebook's own range, for the report."""
        c = C.detach().float()
        out = {"bits": 8, "dtype": "qint8", "step": self.rest[0]}  # "step": the AC grid's, for gn_vq's log line
        for name, (scale, zp), part in (("dc", self.dc, c[:, :N_DC]), ("rest", self.rest, c[:, N_DC:])):
            out[name] = {"scale": scale, "zero_point": zp, "grid_min": (-128 - zp) * scale,
                         "grid_max": (127 - zp) * scale, "codebook_min": float(part.min()),
                         "codebook_max": float(part.max())}
        return out

    def quantize(self, C: Tensor) -> Tuple[Tensor, Tensor, Dict]:
        """``(dequantized centroids, int8 codes, range)``: what C3DGS writes and decodes for these centroids."""
        c = C.detach().float().cpu()
        parts, codes = [], []
        for (scale, zp), part in ((self.dc, c[:, :N_DC]), (self.rest, c[:, N_DC:])):
            q = torch.quantize_per_tensor(part.contiguous(), scale, zp, torch.qint8)
            parts.append(q.dequantize())
            codes.append(q.int_repr())
        return torch.cat(parts, 1).to(C.device), torch.cat(codes, 1), self.range(C)


def run_gn_vq(x: Tensor, C0: Tensor, L0: Tensor, M16, rho: float, total_pixels: int,
              quantizer: C3DGSQuantizer, eps: float = VQ_EPS, max_iters: int = VQ_MAX_ITERS,
              rel_tol: float = VQ_REL_TOL, log: Optional[Callable[[str], None]] = print,
              report_metric=None) -> Tuple[Tensor, Tensor, Dict]:
    """GN-VQ on ``x [n, 48]`` from the warm start ``(C0 [K, 48], L0 [n])`` with ``M16 [n, 136]`` floored at ``rho``;
    returns (float centroids, labels, report). ``report["objectives_under"]["M"]`` is the unfloored objective.

    With ``report_metric`` (the unfloored metric, e.g. a ``metric_store.HostMetric`` in host memory), ``M16`` is taken
    as already floored at ``rho``: a ``metric_store.MetricStore`` buffer, one device copy, filled slice by slice with
    ``e2b.floored_metric`` itself, so the values are those the default path computes (a CPU test checks it bit for
    bit). The default floors here, with a full copy."""
    if M16.shape[1] != gm.D_DC * (gm.D_DC + 1) // 2 or x.shape[1] != N_COLOUR:
        raise ValueError(f"E3r's GN-VQ takes a [n, 136] metric and [n, 48] colours, got {tuple(M16.shape)}, {tuple(x.shape)}")
    t = time.perf_counter()
    Mf, M_rep = (e2b.floored_metric(M16, rho), M16) if report_metric is None else (M16, report_metric)
    C, labels, report = vq.gn_vq(x, C0, L0, Mf, total_pixels, max_iters=max_iters, rel_tol=rel_tol, eps=eps,
                                 topk=min(64, int(C0.shape[0])), log=log, report_metrics={"M": (M_rep, total_pixels)},
                                 quantizer=quantizer)
    report.update(rho=rho, floor="M_i + rho * tr(M_i) / 16 * I", metric_dim=gm.D_DC, n_splats=int(x.shape[0]),
                  n_clusters=int(C0.shape[0]), time_s=time.perf_counter() - t)
    return C, labels, report


def colours_of(sh0: Tensor, shN: Tensor) -> Tensor:
    """``[n, 48]`` in C3DGS's layout (index ``k * 3 + channel``) from gsplat's ``sh0 [n, 1, 3]``, ``shN [n, 15, 3]``."""
    return torch.cat([sh0, shN], dim=1).reshape(sh0.shape[0], N_COLOUR)


def split_colours(c: Tensor) -> Tuple[Tensor, Tensor]:
    """``(sh0 [n, 1, 3], shN [n, 15, 3])`` from ``[n, 48]``."""
    c3 = c.reshape(c.shape[0], 16, 3)
    return c3[:, :1].contiguous(), c3[:, 1:].contiguous()


def colour_dmse(render_rgb: Callable, views: Iterable[Dict], ref: Dict[str, Tensor],
                variants: Dict[str, Tuple[Tensor, Tensor]]) -> Dict[str, Dict]:
    """``diagnostics.measure_dmse``'s formula (mean squared difference over pixels and the 3 channels, clamped to
    [0, 1] as the eval does, and unclamped) with each variant's ``(sh0, shN)`` replacing the reference's."""
    sums = {k: {"clamped": 0.0, "raw": 0.0} for k in variants}
    n_values, n_views = 0, 0
    with torch.no_grad():
        for view in views:
            r = render_rgb(view, ref)
            rc = r.clamp(0.0, 1.0)
            n_values += r.numel()
            n_views += 1
            for name, (sh0, shn) in variants.items():
                img = render_rgb(view, {**ref, "sh0": sh0, "shN": shn})
                sums[name]["raw"] += float((img - r).double().pow(2).sum())
                sums[name]["clamped"] += float((img.clamp(0.0, 1.0) - rc).double().pow(2).sum())
    return {k: {"clamped": s["clamped"] / max(n_values, 1), "raw": s["raw"] / max(n_values, 1),
                "n_values": n_values, "n_views": n_views} for k, s in sums.items()}


def predicted_colour_dmse(M16: Tensor, delta48: Tensor, total_pixels: int) -> float:
    """``P = sum_i sum_rgb delta_i^T M_i delta_i / (3 * pixels)`` over the rows given, ``delta48 [n, 48]``."""
    return gd.predicted_dmse(M16, delta48.reshape(delta48.shape[0], 16, 3), total_pixels)


def trace_shares(M16: Tensor, ids: Tensor, chunk: int = 1 << 18) -> Dict:
    """The rows ``ids``' share of the total trace over all rows of ``M16 [N, 136]``, under the 16 x 16 metric and
    under its 15 x 15 block (bands 1-3, the frozen metric), in float64."""
    tr16 = torch.empty(M16.shape[0], dtype=torch.float64)
    tr15 = torch.empty(M16.shape[0], dtype=torch.float64)
    for s in range(0, M16.shape[0], chunk):
        m = M16[s:s + chunk].double().cpu()
        tr16[s:s + chunk] = gm.trace_packed(m)
        tr15[s:s + chunk] = gm.trace_packed(gm.ac_block(m))
    ids = ids.cpu().long()
    out = {"n_rows": int(M16.shape[0]), "n_selected": int(ids.numel())}
    for name, tr in (("16x16", tr16), ("15x15", tr15)):
        tot, sel = float(tr.sum()), float(tr[ids].sum())
        out[name] = {"total_trace": tot, "selected_trace": sel, "share": sel / tot if tot > 0 else float("nan")}
    return out

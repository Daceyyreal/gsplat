"""E5p's and E5's fork in C3DGS's process (kaggle/PREREG_GN.md Amendment 17 b), installed by
``e3q_c3dgs_run.py --fork CONFIG`` when the config says ``"e5": true`` (``e4p_hooks.install`` picks this class). C3DGS's
source is not edited.

It is E4p's fork (``e4p_hooks.ForkHooks``: C3DGS's own colour VQ first and unchanged as row ``c3dgs``; one geometry
VQ; the state copied when compression returns; every row installed into a restore of that copy, checked, hashed and
saved before any row is evaluated; C3DGS's evaluation of each; fine-tuning from the copy), with E5's rows computed at the
colour call from the same quantizer input, quantized set and 16 x 16 metric:

- ``ogc_plain``, ``ogc_scalar``, ``ogc_gram``: OGC's ``vq.gram_kmeans`` from the verified copy under its three metrics
  (``"plain"``, ``"scalar"``, ``"gram"``), ``lam`` 1e-3, 15 iterations, seed 0, chunk 25,000 (``bench/gn/e5.py``); with
  the config's ``ogc.impl`` "derived" (no verified copy, Amendment 18 e), ``bench/gn/ogc_derived.gram_kmeans_ours``
  computes them with the same settings, and each row records its ``impl``;
- fine-tuning of the rows the config lists (``c3dgs`` and ``ogc_gram`` in the j = 0 processes), each from its own table.

Also recorded: every row's table range against its int8 grid and the quantizer state at every save (E4q's); each OGC
row's time, GPU peaks and host RSS at its start and peak (``cost``; Amendment 17 h) and the host bytes of OGC's second
metric copy for ``"plain"`` and ``"scalar"`` (``vq.py:29``). No GN-VQ (Amendment 17 a).
"""

import os
import sys

import torch

import e4p_hooks

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import_e5():
    saved = list(sys.path)
    sys.path.insert(0, os.path.join(REPO, "bench", "gn"))
    try:
        import e4q
        import e5
        import ogc_derived
    finally:
        sys.path[:] = saved
    return e4q, e5, ogc_derived


class E5Hooks(e4p_hooks.ForkHooks):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.e4q, self.e5, self.od = _import_e5()
        if self.cfg is not None:
            self.fork_rows = tuple(r for r in self.cfg["rows"] if r != "c3dgs")
            unknown = [r for r in self.fork_rows if r not in self.e5.ROW_METRIC]
            if unknown:
                raise ValueError(f"{unknown} are not E5 rows {self.e5.ROWS} (Amendment 17 b)")
            self.rep["fork"]["e5"] = True

    def _ft_source(self, row: str) -> str:
        return row  # E5's fine-tuned rows start from their own tables; there is no alias

    def _row_label(self, row: str) -> str:
        return row

    def _compute_rows(self) -> None:
        cfg, s, e4q, e5 = self.cfg, self.state, self.e4q, self.e5
        x, C0 = s["x"], s["C0"]
        ids = s["vq_ids"]
        K = int(C0.shape[0])
        qa = self.rep["qa_at_colour_vq"]
        q = self.e3r.C3DGSQuantizer.from_state(qa)
        payload = torch.load(cfg["m_path"], map_location="cpu", weights_only=False)
        M = payload.pop("M_packed")
        if M.shape[0] != s["n_ckpt"]:
            raise RuntimeError(f"the metric has {M.shape[0]} rows, the checkpoint {s['n_ckpt']} splats")
        fr = self.rep["fork"]
        fr.update(n_colour_quantized=int(ids.numel()), total_pixels=int(payload["total_pixels"]), metric_path=cfg["m_path"],
                  quantizer_at_colour_vq=qa, threshold=cfg.get("threshold"), j=cfg.get("j"))
        fr["rows"]["c3dgs"]["table_range"] = e4q.table_range(C0, q)
        M_rows = M.index_select(0, ids)  # host: OGC's G
        del M, payload
        oc = cfg.get("ogc") or {}
        device = oc.get("device", "cuda" if self.cuda else "cpu")
        chunk = int(oc.get("chunk", e5.OGC_CHUNK))
        lam = float(oc.get("lam", e5.LAM))
        impl = oc.get("impl", "ogc")
        if impl not in e5.IMPLS:
            raise ValueError(f"ogc.impl {impl!r} is not one of {e5.IMPLS} (Amendment 18 c, e)")
        fr["ogc_impl"] = impl
        vqmod = None
        for row in self.fork_rows:
            rec = fr["rows"].setdefault(row, {"row": row})
            metric = e5.ROW_METRIC[row]
            ours = impl == "derived"
            try:
                with self.cost(row) as c:
                    if ours:
                        C, L, info = self.od.gram_kmeans_ours(x, M_rows, K, lam, device, chunk=chunk, metric=metric)
                    else:
                        if vqmod is None:
                            vqmod = self.ogc.load_vq(oc["clone"])
                        C, L, info = self.ogc.ogc_codebook(vqmod, x, M_rows, K, lam, device, chunk=chunk, metric=metric)
                    rec.update(ogc=info, metric=metric, impl="ogc_derived" if ours else "ogc")
                    c["host_bytes_metric_copy"] = info.get("host_bytes_metric_copy")
                self.tables[row] = (C.detach().to(device=C0.device, dtype=C0.dtype),
                                    L.detach().to(device=s["L0"].device, dtype=s["L0"].dtype))
                rec.update(status="ok", K=int(C.shape[0]), n=int(L.shape[0]), table_range=e4q.table_range(C, q),
                           codewords_used=int((torch.bincount(L.long().cpu(), minlength=K) > 0).sum()))
            except Exception as e:  # noqa: BLE001 (recorded; the job applies the status rule and the attempt rules)
                self._row_error(row, e)
            self.flush()
        del M_rows
        fr["phase"] = "rows_computed"
        self.flush()

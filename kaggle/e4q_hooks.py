"""E4q's fork in C3DGS's process (kaggle/PREREG_GN.md Amendment 16 b-c), installed by ``e3q_c3dgs_run.py --fork CONFIG``
when the config says ``"e4q": true`` (``e4p_hooks.install`` picks this class). C3DGS's source is not edited.

It is E4p's fork (``e4p_hooks.ForkHooks``: C3DGS's own colour VQ first and unchanged as row 1; one geometry VQ; the
state copied when compression returns; every row installed into a restore of that copy, checked, hashed and saved
before any row is evaluated; C3DGS's evaluation of each), with E4q's rows computed at the colour call from the same
quantizer input, quantized set and 16 x 16 metric:

- ``gnvq_cv``: the frozen GN-VQ at ``rho_cv`` (E4p's row 5);
- the ladder and ``lad_all`` (``bench/gn/e4q.ladder_spec``): GN-VQ's loop with OGC's choices, one at a time and all at
  once; the starts drawn as OGC draws (``e4q.ogc_init``) from the unfloored metric of the quantized splats in host
  memory, their starting labels one exact assignment;
- ``ogc`` (``lam`` 1e-3) and ``ogc_lamcv`` (``lam_cv``; not run again when it is 1e-3): OGC's ``gram_kmeans`` from the
  pinned clone at chunk 25,000 (Amendment 16 c);
- at the sweep's points only the rows the config lists (``c3dgs``, ``gnvq_cv``, ``ogc``, ``ogc_lamcv`` and the best
  ladder row).

Also recorded (Amendment 16 b-c): every row's table range against its int8 grid and the values outside it; the
quantizer state at every save (``e4p_hooks``); ``lad_all``'s labels against ``ogc``'s; with ``chunk_check``, OGC's
labels at chunk 100,000 against 25,000 on the same inputs (not saved, not evaluated); the process's reserved peak folded
over the per-row resets. No fine-tuning.
"""

import os
import sys
from typing import Dict

import torch

import e4p_hooks

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import_e4q():
    saved = list(sys.path)
    sys.path.insert(0, os.path.join(REPO, "bench", "gn"))
    try:
        import e4q
    finally:
        sys.path[:] = saved
    return e4q


def _json_spec(spec: Dict) -> Dict:
    return {k: (str(v) if isinstance(v, float) and v == float("-inf") else v) for k, v in spec.items()}


class E4qHooks(e4p_hooks.ForkHooks):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.e4q = _import_e4q()
        if self.cfg is not None:
            self.fork_rows = tuple(r for r in self.cfg["rows"] if r != "c3dgs")
            self.rep["fork"]["e4q"] = True

    def _compute_rows(self) -> None:
        cfg, s, e4q = self.cfg, self.state, self.e4q
        x, C0, L0 = s["x"], s["C0"], s["L0"]
        ids = s["vq_ids"]
        K = int(C0.shape[0])
        qa = self.rep["qa_at_colour_vq"]
        q = self.e3r.C3DGSQuantizer.from_state(qa)
        payload = torch.load(cfg["m_path"], map_location="cpu", weights_only=False)
        M = payload.pop("M_packed")
        if M.shape[0] != s["n_ckpt"]:
            raise RuntimeError(f"the metric has {M.shape[0]} rows, the checkpoint {s['n_ckpt']} splats")
        total_pixels = int(payload["total_pixels"])
        rho_cv, lam_cv = cfg.get("rho_cv"), cfg.get("lam_cv")
        fr = self.rep["fork"]
        fr.update(n_colour_quantized=int(ids.numel()), total_pixels=total_pixels, metric_path=cfg["m_path"],
                  quantizer_at_colour_vq=qa, threshold=cfg.get("threshold"), j=cfg.get("j"))
        fr["rows"]["c3dgs"]["table_range"] = e4q.table_range(C0, q)
        store = self.ms.MetricStore(x.device, ids)
        store.add("full", M)
        M_rows = M.index_select(0, ids)  # host: OGC's G and the draw's weights
        del M
        oc = cfg.get("ogc") or {}
        device = oc.get("device", "cuda" if self.cuda else "cpu")
        chunk = int(oc.get("chunk", e4q.OGC_CHUNK))
        vqmod = None
        xf = x.float()
        for row in self.fork_rows:
            rec = fr["rows"].setdefault(row, {"row": row})
            if row == "ogc_lamcv" and e4q.lamcv_alias(lam_cv):
                rec.update(status="alias", alias_of="ogc", reason="lam_cv = 1e-3: ogc_lamcv is ogc and is not run again "
                                                                  "(Amendment 16 b)")
                continue
            try:
                with self.cost(row) as c:
                    if row in ("ogc", "ogc_lamcv"):
                        if vqmod is None:
                            vqmod = self.ogc.load_vq(oc["clone"])
                        lam = None if row == "ogc" else float(lam_cv)
                        C, L, info = self.ogc.ogc_codebook(vqmod, x, M_rows, K, lam, device, chunk=chunk)
                        rec.update(ogc=info)
                    else:
                        spec = e4q.ladder_spec(row)
                        rho = float(rho_cv) if spec["rho"] == "cv" else 0.0
                        Mf = store.floored("full", rho)
                        if spec["start"] == "ogc_init":
                            Cs, drawn = e4q.ogc_init(xf, M_rows, K, seed=0)
                            Ls = e4q.start_labels(xf, Mf, Cs)
                            rec["start"] = {"drawn": int(drawn.numel()), "first_ids": drawn[:8].tolist(),
                                            "how": "OGC's draw (vq.py:21, 31-33), labels from one exact assignment"}
                        else:
                            Cs, Ls = C0.float(), L0.long()
                        opts = e4q.ladder_options(spec, xf, Mf, C0.float())
                        C, L, rep = self.e3r.run_gn_vq(xf, Cs, Ls, Mf, rho, total_pixels, q, log=self.log,
                                                       report_metric=store.sorted("full"), **opts)
                        rec.update(gn_vq=self._vq_report(rep), rho=rho, spec=_json_spec(spec))
                    c["metric_device_bytes"] = store.device_bytes()
                self.tables[row] = (C.detach().to(device=C0.device, dtype=C0.dtype),
                                    L.detach().to(device=L0.device, dtype=L0.dtype))
                rec.update(status="ok", K=int(C.shape[0]), n=int(L.shape[0]), table_range=e4q.table_range(C, q))
            except Exception as e:  # noqa: BLE001 (recorded; E4q has no primary row)
                self._row_error(row, e)
            store.release()
            self.flush()
        if e4q.LAD_ALL in self.tables and "ogc" in self.tables:
            (Ca, La), (Co, Lo) = self.tables[e4q.LAD_ALL], self.tables["ogc"]
            fr["lad_all_vs_ogc"] = {**e4q.labels_agreement(La, Lo, K),
                                    "codebook_max_abs_diff": float((Ca.float() - Co.float()).abs().max())}
        if cfg.get("chunk_check") and "ogc" in self.tables:
            try:
                with self.cost("ogc_chunk_check"):
                    if vqmod is None:
                        vqmod = self.ogc.load_vq(oc["clone"])
                    C100, L100, info = self.ogc.ogc_codebook(vqmod, x, M_rows, K, None, device, chunk=e4q.OGC_CHUNK_CHECK)
                Co, Lo = self.tables["ogc"]
                fr["chunk_check"] = {**e4q.chunk_check(Lo, L100, Co, C100), "call": info["call"]}
            except Exception as e:  # noqa: BLE001 (report only)
                fr["chunk_check"] = {"error": f"{type(e).__name__}: {str(e)[:600]}"}
        del M_rows
        store.release()
        fr["phase"] = "rows_computed"
        self.flush()

"""E3r's hooks in C3DGS's process (kaggle/PREREG_GN.md Amendment 14 b-c), installed by ``e3q_c3dgs_run.py`` before it
runs ``compress.py`` unchanged. C3DGS's source is not edited.

- **observe** (every E3r run): counts the splats C3DGS prunes, keeps with their own colour and colour-quantizes;
  records its colour fake quantizers' scale and zero point when its colour VQ runs and when it writes the ``.npz``,
  and a SHA-1 of its geometry table after the covariance VQ (report only: a probe run and an injected run should
  hold the same one). It reads tensors only: it never calls ``get_features``, whose fake quantizers update their
  observers, so what C3DGS computes does not change.
- **record** (the probe run): also writes the colour VQ's input (``[n, 48]``), the checkpoint ids of the quantized
  and kept splats, the kept rows, C3DGS's codebook and labels, and the quantizer state, for the harness's
  cross-validation and the trace share.
- **inject** (the GN-VQ runs): C3DGS's own ``vq_features`` runs first, unchanged (the global random streams advance
  as in its own run, and its codebook and labels are the warm start); then GN-VQ with the floored 16 x 16 metric at
  ``rho_cv`` (``bench/gn/e3r.run_gn_vq``, C3DGS's int8 table quantizer) replaces them. Also recorded: the
  difference from the probe run's quantized set, the report-only lifted check, and, when C3DGS writes the
  ``.npz``, whether its colour indices still equal the injected ones and how far its table moved.

Patching ``compression.vq.compress_gaussians`` reaches ``compress.py``, which imports it by name when ``runpy``
runs it; ``compress_color``, ``vq_features`` and ``compress_covariance`` are called through the module's globals.
"""

import hashlib
import inspect
import json
import os
import sys
import time
import traceback
from typing import Dict, Optional

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _qa_state(g) -> Dict:
    """The colour fake quantizers' scale and zero point (buffers; reading them has no side effect)."""
    return {"dc_scale": float(g.features_dc_qa.scale.reshape(-1)[0]),
            "dc_zero_point": int(g.features_dc_qa.zero_point.reshape(-1)[0]),
            "rest_scale": float(g.features_rest_qa.scale.reshape(-1)[0]),
            "rest_zero_point": int(g.features_rest_qa.zero_point.reshape(-1)[0])}


def _table(g) -> torch.Tensor:
    """The colour table ``[T, 48]`` as C3DGS holds it (raw parameters, no fake quantization applied)."""
    return torch.cat([g._features_dc.detach(), g._features_rest.detach()], dim=1).reshape(g._features_dc.shape[0], -1)


def _import_gn():
    saved = list(sys.path)
    sys.path.insert(0, os.path.join(REPO, "bench", "gn"))
    try:
        import diagnostics as gd
        import e2b
        import e3r
    finally:
        sys.path[:] = saved
    return gd, e2b, e3r


class Hooks:
    def __init__(self, record: Optional[str] = None, inject: Optional[str] = None, log=print):
        self.record_path, self.log = record, log
        self.inject_cfg = json.load(open(inject)) if inject else None
        self.state: Dict = {"in_colour": False}
        self.rep: Dict = {"mode": "inject" if inject else "record" if record else "observe", "errors": []}
        self.orig: Dict = {}

    # ------------------------------------------------------------------ installation
    def install(self) -> "Hooks":
        import compression.vq as cvq
        from scene.gaussian_model import GaussianModel

        self.cvq, self.GM = cvq, GaussianModel
        for name in ("compress_gaussians", "compress_color", "vq_features", "compress_covariance"):
            self.orig[name] = getattr(cvq, name)
        self.orig["save_npz"] = GaussianModel.save_npz
        self.sig = {n: inspect.signature(self.orig[n]) for n in ("compress_gaussians", "compress_color", "vq_features")}
        cvq.compress_gaussians = self.compress_gaussians
        cvq.compress_color = self.compress_color
        cvq.vq_features = self.vq_features
        cvq.compress_covariance = self.compress_covariance
        hooks = self

        def save_npz(model, *args, **kwargs):
            hooks.before_save(model)
            return hooks.orig["save_npz"](model, *args, **kwargs)

        GaussianModel.save_npz = save_npz
        return self

    def _bind(self, name, args, kwargs):
        b = self.sig[name].bind(*args, **kwargs)
        b.apply_defaults()
        return b.arguments

    def _safe(self, what, fn):
        try:
            fn()
        except Exception as e:  # recording must not change or stop C3DGS's own run
            self.rep["errors"].append({"where": what, "error": f"{type(e).__name__}: {str(e)[:400]}",
                                       "traceback": traceback.format_exc()[-2000:]})

    # ------------------------------------------------------------------ the wrapped functions
    def compress_gaussians(self, *args, **kwargs):
        a = self._bind("compress_gaussians", args, kwargs)
        g, ci, pt = a["gaussians"], a["color_importance"], a["prune_threshold"]
        n = int(g._xyz.shape[0])
        mask = (ci > pt) if pt >= 0 else torch.ones(n, dtype=torch.bool, device=ci.device)
        self.state.update(n_ckpt=n, non_prune=mask.detach().cpu(), prune_threshold=float(pt))
        self.rep["counts"] = {"n_ckpt": n, "n_pruned": n - int(mask.sum()), "n_after_prune": int(mask.sum()),
                              "prune_threshold": float(pt)}
        return self.orig["compress_gaussians"](*args, **kwargs)

    def compress_color(self, *args, **kwargs):
        a = self._bind("compress_color", args, kwargs)
        g, ci, comp = a["gaussians"], a["color_importance"], a["color_comp"]
        keep = (ci > comp.importance_include).detach().cpu()
        after = self.state["non_prune"].nonzero(as_tuple=True)[0]
        self.state.update(gaussians=g, keep=keep, vq_ids=after[~keep], kept_ids=after[keep], K=int(comp.codebook_size))
        self.rep["counts"].update(n_kept_colour=int(keep.sum()), n_colour_quantized=int((~keep).sum()),
                                  color_codebook_size=int(comp.codebook_size),
                                  color_importance_include=float(comp.importance_include),
                                  color_compress_non_dir=bool(a["color_compress_non_dir"]))
        self.state["in_colour"] = True
        try:
            out = self.orig["compress_color"](*args, **kwargs)
        finally:
            self.state["in_colour"] = False
        self.state["indices_after"] = g._feature_indices.detach().cpu().clone()
        self.state["table_after"] = _table(g).cpu().clone()
        if self.record_path:
            self._safe("record", self._write_record)
        return out

    def vq_features(self, *args, **kwargs):
        C0, L0 = self.orig["vq_features"](*args, **kwargs)  # unchanged, always first: the random streams
        a = self._bind("vq_features", args, kwargs)
        if not self.state["in_colour"] or a["scale_normalize"]:
            return C0, L0
        g = self.state["gaussians"]
        qa = _qa_state(g)
        self.rep["qa_at_colour_vq"] = qa
        self.state.update(x=a["features"].detach(), C0=C0.detach(), L0=L0.detach())
        if self.inject_cfg is None:
            return C0, L0
        C, L = self._inject(a["features"].detach(), C0.detach(), L0.detach(), qa)
        return C.to(device=C0.device, dtype=C0.dtype), L.to(device=L0.device, dtype=L0.dtype)

    def compress_covariance(self, *args, **kwargs):
        out = self.orig["compress_covariance"](*args, **kwargs)

        def geo():
            g = args[0] if args else kwargs["gaussians"]
            h = hashlib.sha1()
            for t in (g._gaussian_indices, g._rotation, g._scaling):
                h.update(t.detach().contiguous().cpu().numpy().tobytes())
            self.rep["geometry_sha1"] = h.hexdigest()

        self._safe("geometry_sha1", geo)
        return out

    def before_save(self, g) -> None:
        def check():
            self.rep["qa_at_save"] = _qa_state(g)
            if self.inject_cfg is None or "indices_after" not in self.state:
                return
            now_idx = g._feature_indices.detach().cpu()
            table = _table(g).cpu()
            K = self.state["K"]
            before = self.state["table_after"]
            self.rep["save_check"] = {
                "labels_survived": bool(torch.equal(now_idx, self.state["indices_after"])),
                "n_labels_changed": int((now_idx != self.state["indices_after"]).sum()) if now_idx.shape == self.state["indices_after"].shape else -1,
                "codebook_max_abs_change": float((table[:K] - before[:K]).abs().max()),
                "kept_rows_max_abs_change": float((table[K:] - before[K:]).abs().max()) if table.shape[0] > K else 0.0,
            }

        self._safe("before_save", check)

    # ------------------------------------------------------------------ record and inject
    def _write_record(self) -> None:
        s = self.state
        K = s["K"]
        payload = {"n_ckpt": s["n_ckpt"], "non_prune": s["non_prune"], "keep": s["keep"], "vq_ids": s["vq_ids"],
                   "kept_ids": s["kept_ids"], "x": s["x"].cpu(), "kept_rows": s["table_after"][K:].clone(),
                   "codebook": s["C0"].cpu(), "labels": s["L0"].cpu(), "qa": self.rep["qa_at_colour_vq"], "K": K,
                   "threshold": self.rep["counts"]["color_importance_include"]}
        os.makedirs(os.path.dirname(os.path.abspath(self.record_path)), exist_ok=True)
        torch.save(payload, self.record_path + ".tmp")
        os.replace(self.record_path + ".tmp", self.record_path)
        self.rep["record"] = {"path": self.record_path, "bytes": os.path.getsize(self.record_path),
                              "n_colour_quantized": int(s["vq_ids"].numel()), "n_kept_colour": int(s["kept_ids"].numel())}

    def _inject(self, x, C0, L0, qa):
        gd, e2b, e3r = _import_gn()
        cfg = self.inject_cfg
        t0 = time.perf_counter()
        ids = self.state["vq_ids"]
        payload = torch.load(cfg["m_path"], map_location="cpu", weights_only=False)
        M = payload["M_packed"]
        if M.shape[0] != self.state["n_ckpt"]:
            raise RuntimeError(f"the metric has {M.shape[0]} rows, the checkpoint {self.state['n_ckpt']} splats")
        M_sub = M.index_select(0, ids).to(x.device)
        del M, payload["M_packed"]
        rep = {"rho": cfg["rho"], "n_colour_quantized": int(ids.numel()), "metric_path": cfg["m_path"],
               "total_pixels": int(payload["total_pixels"]), "quantizer_at_injection": qa}
        if cfg.get("probe_record") and os.path.exists(cfg["probe_record"]):
            probe = torch.load(cfg["probe_record"], map_location="cpu", weights_only=False)
            a, b = set(probe["vq_ids"].tolist()), set(ids.tolist())
            rep["set_vs_probe"] = {"n_probe": len(a), "n_injected": len(b), "only_probe": len(a - b),
                                   "only_injected": len(b - a), "same_set": a == b}
            if a == b and probe["x"].shape == x.shape:
                rep["set_vs_probe"]["features_equal"] = bool(torch.equal(probe["x"], x.cpu()))
            del probe
        q = e3r.C3DGSQuantizer.from_state(qa)
        C, L, report = e3r.run_gn_vq(x.float(), C0.float(), L0.long(), M_sub, float(cfg["rho"]), rep["total_pixels"], q,
                                     eps=cfg.get("eps", e3r.VQ_EPS), max_iters=cfg.get("max_iters", e3r.VQ_MAX_ITERS),
                                     rel_tol=cfg.get("rel_tol", e3r.VQ_REL_TOL), log=self.log)
        rep["gn_vq"] = {k: v for k, v in report.items() if k != "history"}
        rep["gn_vq"]["history"] = report["history"][-3:]
        if cfg.get("n_lifted_check", 10000):
            def lifted():
                Mf = e2b.floored_metric(M_sub, float(cfg["rho"]))
                Cq = q.quantize(C)[0]
                rep["lifted_check"] = gd.lifted_check(x.float(), Mf, Cq, n_sample=int(cfg.get("n_lifted_check", 10000)), seed=0)

            self._safe("lifted_check", lifted)
        del M_sub
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rep["time_s"] = time.perf_counter() - t0
        self.rep["inject"] = rep
        return C, L

    def report(self) -> Dict:
        return self.rep


def install(record: Optional[str] = None, inject: Optional[str] = None) -> Hooks:
    return Hooks(record=record, inject=inject).install()

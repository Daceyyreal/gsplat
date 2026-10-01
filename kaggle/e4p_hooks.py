"""E4p's fork in C3DGS's process (kaggle/PREREG_GN.md Amendment 15 b, d, e and note ii e), installed by
``e3q_c3dgs_run.py`` before it runs ``compress.py`` unchanged. C3DGS's source is not edited. It extends E3r's hooks
(``e3r_hooks.Hooks``: the counts, the quantizer state, the geometry SHA-1, the probe record).

Three modes:

- **probe** (``--record PATH --defer_eval``): E3r's record of the colour VQ, and C3DGS's evaluation deferred: at the
  save, ``compress.py``'s module-global ``render_and_eval`` (looked up at call time, ``compress.py:242``) is replaced
  in ``run_vq``'s globals by a function that evaluates nothing (Amendment 15 d: the probe is evaluated later, from
  its decoded ``.npz``).
- **fork** (``--fork CONFIG``), Amendment 15 b:
  1. C3DGS's own ``vq_features`` runs first, unchanged; its codebook and labels go back to C3DGS (row 1).
  2. At that colour call, rows 2, 2b, 3, 4 and 5 are computed from the same quantizer input, quantized set and
     16 x 16 metric (row 5 not when ``rho_cv`` = 0: row 3 stands for it). ``join_features``, which
     ``compress_color`` calls through the module's globals, is wrapped to capture its exact ``all_features`` and
     ``keep_mask``, so each row is installed with C3DGS's own ``join_features`` and ``set_color_indexed``.
  3. C3DGS's geometry VQ runs once.
  4. When ``compress_gaussians`` returns, the model (every attribute: parameters, the fake quantizers' state with
     their observers) and the random states are copied to host memory (``e4p.copy_state``). Every other row is
     installed into a restore of **that** copy, never into a saved (Morton-sorted) state.
  5. At the save (``compress.py:231``), row 1's checks; ``run_vq``'s frame gives ``scene`` and the parameters; its
     ``render_and_eval`` is replaced by the fork's phase, which C3DGS then calls at ``compress.py:242``:
     (a) every other row restored, installed, checked and saved (``save_npz``, which sorts in place), each saved
     state copied; (b) only then every row evaluated with C3DGS's ``render_and_eval``, each from its own saved state;
     (c) fine-tuning of rows 1, 2 and 5 (Amendment 15 e): restored from the copy with the copy's random states, the
     row installed, C3DGS's ``finetune`` with 5,000 iterations, saved, evaluated; (d) row 1's saved state put back,
     and row 1's metrics returned, so ``compress.py`` writes ``results.json`` as usual.
  The checks per row (a failure is a bug; the process's rows enter nothing): the keep mask and quantized set equal
  row 1's, the geometry SHA-1 equal at its save, the colour indices just before its save equal its labels. Each row's
  installed state just before its save (and each fine-tuned row's before fine-tuning) is hashed (``e4p.state_sha1``),
  so a test can show it equals the same row installed into an untouched model, in the pre-sort order. Each row's
  time, peak GPU memory and the process's host RSS (with its children) are recorded (note ii e). The report is
  written to ``report_path`` after every row, so a crash keeps what finished.
- **eval_npz** (``--eval_npz NPZ``): C3DGS's evaluation of a decoded ``.npz`` (the probe, Amendment 15 d).
  ``scene.Scene`` is replaced by a subclass before ``compress.py`` imports it; once C3DGS's own ``Scene`` has loaded
  the model, the ``.npz`` is loaded into it (C3DGS's ``load_npz``), C3DGS's ``render_and_eval`` runs, and the run
  stops there (``Stop``, which the wrapper records as a success).
"""

import copy
import hashlib
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from typing import Dict, Optional

import torch

import e3r_hooks

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import_e4p():
    saved = list(sys.path)
    sys.path.insert(0, os.path.join(REPO, "bench", "gn"))
    sys.path.insert(0, os.path.join(REPO, "kaggle"))
    try:
        import e3r
        import e4p
        import e4p_ogc
        import gn_metric as gm
        import metric_store as ms
    finally:
        sys.path[:] = saved
    return e3r, e4p, e4p_ogc, gm, ms


class Stop(Exception):
    """Raised after ``--eval_npz``'s evaluation; the wrapper records the run as a success."""


def _geometry_sha1(g) -> str:
    h = hashlib.sha1()
    for t in (g._gaussian_indices, g._rotation, g._scaling):
        h.update(t.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def _find_frame(names):
    f = sys._getframe(1)
    while f is not None:
        if all(n in f.f_locals for n in names):
            return f
        f = f.f_back
    return None


class ForkHooks(e3r_hooks.Hooks):
    def __init__(self, record: Optional[str] = None, fork: Optional[str] = None, defer_eval: bool = False,
                 eval_npz: Optional[str] = None, log=print):
        super().__init__(record=record, inject=None, log=log)
        self.e3r, self.e4p, self.ogc, self.gm, self.ms = _import_e4p()
        self.cfg = json.load(open(fork)) if fork else None
        self.defer_eval = bool(defer_eval or fork)
        self.eval_npz = eval_npz
        self.rep["mode"] = "fork" if fork else "eval_npz" if eval_npz else "probe_deferred" if defer_eval else self.rep["mode"]
        self.cuda = torch.cuda.is_available()
        self.cuda_peak = 0
        self.stopped_ok = False
        self.tables: Dict[str, tuple] = {}
        if self.cfg is not None:
            self.rep["fork"] = {"rows": {}, "cost": {}, "checks_failed": [], "phase": "start",
                                "rho_cv": self.cfg.get("rho_cv"), "config": {k: v for k, v in self.cfg.items()}}

    # ------------------------------------------------------------------ installation
    def install(self) -> "ForkHooks":
        super().install()
        self.orig["join_features"] = self.cvq.join_features
        self.cvq.join_features = self.join_features
        if self.eval_npz:
            import scene as scene_mod

            hooks, Orig = self, scene_mod.Scene

            class E4pEvalScene(Orig):
                def __init__(s, *a, **k):
                    super().__init__(*a, **k)
                    hooks._evaluate_npz(s)

            scene_mod.Scene = E4pEvalScene
        return self

    # ------------------------------------------------------------------ bookkeeping
    def fold_peak(self) -> int:
        if self.cuda:
            self.cuda_peak = max(self.cuda_peak, torch.cuda.max_memory_allocated())
        return self.cuda_peak

    @contextmanager
    def cost(self, name: str):
        """Time, peak allocated GPU memory and host RSS (process and children) of one step (note ii e)."""
        if self.cuda:
            torch.cuda.synchronize()
            self.fold_peak()
            torch.cuda.reset_peak_memory_stats()
        rec = {"start_allocated": torch.cuda.memory_allocated() if self.cuda else None}
        t = time.perf_counter()
        h = self.e4p.HostRss().__enter__()
        try:
            yield rec
        finally:  # recorded for a failed step too
            if self.cuda:
                torch.cuda.synchronize()
            rec["time_s"] = time.perf_counter() - t
            rec["cuda_peak_allocated"] = torch.cuda.max_memory_allocated() if self.cuda else None
            self.fold_peak()
            h.__exit__(None, None, None)
            rec["host"] = h.record()
            if self.cfg is not None:
                self.rep["fork"]["cost"][name] = rec

    def flush(self) -> None:
        if self.cfg is not None and self.cfg.get("report_path"):
            p = self.cfg["report_path"]
            with open(p + ".tmp", "w") as f:
                json.dump(self.rep, f, indent=1, default=str)
            os.replace(p + ".tmp", p)

    def _row_error(self, row: str, e: BaseException) -> None:
        self.rep["fork"]["rows"].setdefault(row, {}).update(
            status="failed", error=f"{type(e).__name__}: {str(e)[:600]}", oom=self.e4p.is_oom_text(f"{type(e).__name__}: {e}"),
            traceback=traceback.format_exc()[-3000:])
        if self.cuda:
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ the wrapped functions
    def join_features(self, all_features, keep_mask, codebook, codebook_indices):
        if self.state.get("in_colour"):
            self.state["join_inputs"] = (all_features.detach(), keep_mask.detach())
        return self.orig["join_features"](all_features, keep_mask, codebook, codebook_indices)

    def compress_gaussians(self, *args, **kwargs):
        out = super().compress_gaussians(*args, **kwargs)
        if self.cfg is not None:
            g = self._bind("compress_gaussians", args, kwargs)["gaussians"]
            self.copy = {"model": self.e4p.copy_state(g), "rng": self.e4p.rng_state()}
            self.rep["fork"]["copy"] = {"taken": "when compress_gaussians returned (Amendment 15 b.4)",
                                        "host_bytes": self.e4p.copy_bytes(self.copy["model"]),
                                        "attributes": self.copy["model"]["keys"]}
            self.rep["fork"]["phase"] = "compressed"
            self.flush()
        return out

    def vq_features(self, *args, **kwargs):
        a = self._bind("vq_features", args, kwargs)
        colour = self.state.get("in_colour") and not a["scale_normalize"]
        if self.cfg is None or not colour:
            return super().vq_features(*args, **kwargs)
        with self.cost("c3dgs"):
            C0, L0 = super().vq_features(*args, **kwargs)  # C3DGS's own VQ, unchanged and first
        self.tables["c3dgs"] = (C0.detach(), L0.detach())
        self.rep["fork"]["rows"]["c3dgs"] = {"status": "ok", "K": int(C0.shape[0]), "n": int(L0.shape[0]), "row": "1"}
        self._compute_rows()
        return C0, L0

    def _compute_rows(self) -> None:
        cfg, s = self.cfg, self.state
        x, C0, L0 = s["x"], s["C0"], s["L0"]
        ids = s["vq_ids"]
        qa = self.rep["qa_at_colour_vq"]
        q = self.e3r.C3DGSQuantizer.from_state(qa)
        payload = torch.load(cfg["m_path"], map_location="cpu", weights_only=False)
        M = payload.pop("M_packed")
        if M.shape[0] != s["n_ckpt"]:
            raise RuntimeError(f"the metric has {M.shape[0]} rows, the checkpoint {s['n_ckpt']} splats")
        total_pixels = int(payload["total_pixels"])
        rho_cv = cfg.get("rho_cv")
        fr = self.rep["fork"]
        fr.update(n_colour_quantized=int(ids.numel()), total_pixels=total_pixels, metric_path=cfg["m_path"],
                  quantizer_at_colour_vq=qa)
        store = self.ms.MetricStore(x.device, ids)
        store.add("full", M)
        M_rows = None
        vqmod = None
        for row in self.e4p.FORK_ROWS:
            rec = fr["rows"].setdefault(row, {"row": self.e4p.ROW_NUMBER[row]})
            if row == "gnvq_cv" and rho_cv == 0:
                rec.update(status="alias", alias_of="gnvq_rho0",
                           reason="rho_cv = 0: row 5 is row 3 by definition and is not run again (Amendment 15 b)")
                continue
            try:
                with self.cost(row) as c:
                    if row in ("ogc", "ogc_lam1e6"):
                        if vqmod is None:
                            vqmod = self.ogc.load_vq(cfg["ogc"]["clone"])
                        if M_rows is None:
                            M_rows = M.index_select(0, ids)
                        C, L, info = self.ogc.ogc_codebook(vqmod, x, M_rows, int(C0.shape[0]), self.e4p.OGC_LAM[row],
                                                           cfg["ogc"].get("device", "cuda" if self.cuda else "cpu"))
                        rec.update(ogc=info)
                    elif row == "scalar":
                        iso = self.ms.MetricStore(x.device, torch.arange(ids.numel()))
                        iso.add("iso", self.e4p.isotropic_packed(M.index_select(0, ids) if M_rows is None else M_rows))
                        C, L, rep = self.e3r.run_gn_vq(x.float(), C0.float(), L0.long(), iso.floored("iso", 0.0), 0.0,
                                                       total_pixels, q, log=self.log, report_metric=iso.sorted("iso"))
                        iso.release()
                        rec.update(gn_vq=self._vq_report(rep), metric="tr(M_i) / 16 * I")
                    else:
                        rho = 0.0 if row == "gnvq_rho0" else float(rho_cv)
                        C, L, rep = self.e3r.run_gn_vq(x.float(), C0.float(), L0.long(), store.floored("full", rho), rho,
                                                       total_pixels, q, log=self.log, report_metric=store.sorted("full"))
                        rec.update(gn_vq=self._vq_report(rep), rho=rho)
                    c["metric_device_bytes"] = store.device_bytes()
                self.tables[row] = (C.detach().to(device=C0.device, dtype=C0.dtype),
                                    L.detach().to(device=L0.device, dtype=L0.dtype))
                rec.update(status="ok", K=int(C.shape[0]), n=int(L.shape[0]))
            except Exception as e:  # noqa: BLE001 (recorded; the job applies Amendment 15 d's rules)
                self._row_error(row, e)
            store.release() if row in ("gnvq_rho0", "gnvq_cv") else None
            self.flush()
        del M, M_rows
        store.release()
        fr["phase"] = "rows_computed"
        self.flush()

    @staticmethod
    def _vq_report(rep: Dict) -> Dict:
        out = {k: v for k, v in rep.items() if k not in ("history",)}
        out["history_last3"] = rep.get("history", [])[-3:]
        return out

    # ------------------------------------------------------------------ the save and the deferred evaluation
    def before_save(self, g) -> None:
        super().before_save(g)
        if self.defer_eval and not self.state.get("patched_eval"):
            f = _find_frame(("scene", "comp_params"))
            if f is None:
                raise RuntimeError("E4p: compress.py's run_vq frame (scene, comp_params) was not found at the save")
            fl = f.f_locals
            self.ctx = {k: fl.get(k) for k in ("scene", "model_params", "optim_params", "pipeline_params", "comp_params")}
            self.orig_eval = f.f_globals["render_and_eval"]
            self.finetune_fn = f.f_globals.get("finetune")
            f.f_globals["render_and_eval"] = self.deferred_eval
            self.state["patched_eval"] = True
            self.rep["render_and_eval"] = "replaced in run_vq's globals at the save (deferred evaluation)"
        if self.cfg is not None and "c3dgs" in self.tables and not self.state.get("row1_checked"):
            self.state["row1_checked"] = True
            self._check("c3dgs", g)
            self.rep["fork"]["rows"]["c3dgs"]["pre_save_sha1"] = self.e4p.state_sha1(g)
            self.rep["fork"]["rows"]["c3dgs"]["npz"] = "compress.py's own output"
            self.flush()

    def _install(self, g, row: str) -> None:
        all_features, keep = self.state["join_inputs"]
        C, L = self.tables[row]
        compressed, indices = self.orig["join_features"](all_features, keep, C, L)
        g.set_color_indexed(compressed.reshape(-1, 16, 3), indices)

    def _check(self, row: str, g) -> bool:
        """Amendment 15 b's per-process checks for one row, just before its save."""
        all_features, keep = self.state["join_inputs"]
        C, L = self.tables[row]
        K = int(C.shape[0])
        idx = g._feature_indices.detach()
        keep_d = keep.to(idx.device)
        table = torch.cat([g._features_dc.detach(), g._features_rest.detach()], dim=1).reshape(g._features_dc.shape[0], -1)
        n_keep = int(keep.sum())
        res = {
            "keep_mask_and_set_equal": bool(idx.shape[0] == keep.shape[0]
                                            and torch.equal(idx[keep_d], torch.arange(n_keep, device=idx.device) + K)
                                            and table.shape[0] == K + n_keep
                                            and torch.equal(table[K:], all_features[keep].reshape(n_keep, -1).to(table.device))),
            "labels_equal": bool(torch.equal(idx[~keep_d], L.to(idx.device).long())),
            "geometry_sha1_equal": _geometry_sha1(g) == self.rep.get("geometry_sha1"),
        }
        res["ok"] = all(res.values())
        self.rep["fork"]["rows"].setdefault(row, {})["checks"] = res
        if not res["ok"]:
            self.rep["fork"]["checks_failed"].append(row)
        return res["ok"]

    def deferred_eval(self, gaussians, scene, model_params, pipeline_params):
        if self.cfg is None:  # the probe: evaluated later, from its decoded .npz (Amendment 15 d)
            self.rep["deferred_eval"] = {"evaluated": False, "reason": "Amendment 15 d: the probe is evaluated after "
                                                                      "the first process, from its decoded .npz"}
            return {"deferred": True}
        g, cfg, fr = gaussians, self.cfg, self.rep["fork"]
        e4p = self.e4p
        comp = self.ctx["comp_params"]
        sort = not getattr(comp, "not_sort_morton", False)
        saved = {"c3dgs": e4p.copy_state(g)}  # row 1's saved state (compress.py saved it)
        # (a) every other row: restored from the copy taken when compression returned, installed, checked, saved
        fr["phase"] = "saving"
        for row in e4p.FORK_ROWS:
            if row not in self.tables:
                continue
            path = os.path.join(cfg["rows_dir"], row, "point_cloud.npz")
            try:
                with self.cost(f"save_{row}"):
                    e4p.restore_state(g, self.copy["model"])
                    self._install(g, row)
                    self._check(row, g)
                    fr["rows"][row]["pre_save_sha1"] = e4p.state_sha1(g)  # the installed state, before the sort
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    self.orig["save_npz"](g, path, sort_morton=sort)
                    saved[row] = e4p.copy_state(g)
                fr["rows"][row]["npz"] = path
            except Exception as e:  # noqa: BLE001
                self._row_error(row, e)
            self.flush()
        # (b) only then, every row evaluated by C3DGS, each from its own saved state
        fr["phase"] = "evaluating"
        for row, st in saved.items():
            try:
                with self.cost(f"eval_{row}"):
                    e4p.restore_state(g, st)
                    m = self.orig_eval(g, scene, model_params, pipeline_params)
                fr["rows"][row]["c3dgs_eval"] = {k: float(v) for k, v in m.items()}
            except Exception as e:  # noqa: BLE001 (C3DGS's own evaluation is a secondary: recorded only)
                fr["rows"][row]["c3dgs_eval_error"] = f"{type(e).__name__}: {str(e)[:600]}"
                fr["rows"][row]["c3dgs_eval_oom"] = e4p.is_oom_text(f"{type(e).__name__}: {e}")
            self.flush()
        # (c) fine-tuning, rows 1, 2 and 5 (Amendment 15 e), each from the copy and the copy's random states
        fr["phase"] = "finetuning"
        ft = cfg.get("finetune") or {}
        for row in ft.get("rows", []):
            src = e4p.row_alias(row, cfg.get("rho_cv"))
            name = e4p.ft_name(row)
            rec = fr["rows"].setdefault(name, {"row": e4p.ROW_NUMBER[row] + " fine-tuned", "table_of": src})
            if src not in self.tables:
                rec.update(status="not_run", reason=f"no table for {src}")
                continue
            path = os.path.join(cfg["rows_dir"], name, "point_cloud.npz")
            try:
                with self.cost(f"finetune_{row}"):
                    e4p.restore_state(g, self.copy["model"])
                    e4p.set_rng_state(self.copy["rng"])
                    self._install(g, src)
                    rec["pre_finetune_sha1"] = e4p.state_sha1(g)
                    comp_ft = copy.copy(comp)
                    comp_ft.finetune_iterations = int(ft.get("iterations", e4p.FINETUNE_ITERATIONS))
                    comp_ft.output_vq = os.path.dirname(path)
                    self.finetune_fn(scene, model_params, self.ctx["optim_params"], comp_ft, pipeline_params,
                                     testing_iterations=[-1], debug_from=-1)
                    _, keep = self.state["join_inputs"]
                    idx = g._feature_indices.detach()
                    rec["labels_survived"] = bool(torch.equal(idx[~keep.to(idx.device)], self.tables[src][1].to(idx.device).long()))
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    self.orig["save_npz"](g, path, sort_morton=sort)
                rec["npz"] = path
                with self.cost(f"eval_{name}"):
                    m = self.orig_eval(g, scene, model_params, pipeline_params)
                rec.update(status="ok", c3dgs_eval={k: float(v) for k, v in m.items()},
                           finetune_iterations=comp_ft.finetune_iterations)
            except Exception as e:  # noqa: BLE001 (a secondary: recorded only)
                self._row_error(name, e)
            self.flush()
        # (d) row 1's saved state back; row 1's metrics for compress.py's results.json
        e4p.restore_state(g, saved["c3dgs"])
        fr["phase"] = "done"
        self.flush()
        return dict(fr["rows"]["c3dgs"].get("c3dgs_eval") or {})

    # ------------------------------------------------------------------ eval_npz
    def _evaluate_npz(self, scene_obj) -> None:
        f = _find_frame(("gaussians", "model_params", "pipeline_params"))
        if f is None:
            raise RuntimeError("E4p: compress.py's run_vq frame was not found at the Scene")
        g = f.f_locals["gaussians"]
        with self.cost("eval_npz") as c:
            g.load_npz(self.eval_npz)
            m = f.f_globals["render_and_eval"](g, scene_obj, f.f_locals["model_params"], f.f_locals["pipeline_params"])
        self.rep["eval_npz"] = {"npz": self.eval_npz, "c3dgs_eval": {k: float(v) for k, v in m.items()}, "cost": c,
                                "how": "C3DGS's Scene, then load_npz of the decoded .npz, then render_and_eval"}
        self.stopped_ok = True
        raise Stop("evaluated the .npz; compress.py stops here")

    def report(self) -> Dict:
        self.fold_peak()
        self.rep["cuda_peak_allocated_process"] = self.cuda_peak if self.cuda else None
        return self.rep


def install(record: Optional[str] = None, fork: Optional[str] = None, defer_eval: bool = False,
            eval_npz: Optional[str] = None) -> ForkHooks:
    return ForkHooks(record=record, fork=fork, defer_eval=defer_eval, eval_npz=eval_npz).install()

"""E7's fork in C3DGS's process (kaggle/PREREG_GN.md Amendment 20 d, e), installed by ``e3q_c3dgs_run.py --fork CONFIG``
when the config says ``"e7": true`` (``e4p_hooks.install`` picks this class). C3DGS's source is not edited.

It is E6 part (b)'s fork (``e5_hooks.E5Hooks``: C3DGS's own colour VQ first and unchanged as ``c3dgs``, ``ogc_gram``
from OGC's code at the colour call, one geometry VQ, the copy when compression returns, every row installed into a
restore of that copy and saved before any row is evaluated, fine-tuning from the copy with the copy's random states),
with E7's rows (``kaggle/e7_scenes.py``):

- **``c3dgs_own`` and ``ogc_gram_own``** (full processes): ``c3dgs``'s and ``ogc_gram``'s tables saved with their own
  grid. ``own_grid`` sets each colour quantizer's scale and zero point (and its observer's minimum and maximum, so an
  evaluation's observer keeps that grid) from the installed table part's own minimum and maximum, by torch's min/max
  qparams for qint8, per-tensor affine (``own_qparams``, computed here: no quantizer module is called). Under those
  qparams one end of the part's range can sit up to half a step past the end code; it rounds to that code, so no value
  is clamped (a test checks it).
- **The fine-tuned arms** (20 d), through E4p's per-row hook (Amendment 20 k). ``ft_before_finetune``, after the row's
  restore and install: for G, CG, K and O ``disable_observer()`` on both colour quantizers (O after setting its own
  grid), so fake quantization stays on against the frozen grid; for CG a gradient hook zeroing the colour table's rows
  ``[:K]`` (the codewords), for K one zeroing rows ``[K:]`` (the kept rows) and ``requires_grad_(False)`` on
  ``_opacity``, ``_scaling``, ``_rotation``, ``_scaling_factor`` and ``_xyz``. C3DGS's ``training_setup`` builds a new
  Adam at each fine-tuning, so zero-gradient rows keep zero moments and an update of exactly 0. The next row's restore
  (``e4p.restore_state``) puts the observer flags (the quantizers' ``state_dict``), ``requires_grad`` and fresh
  parameters (without the hooks) back. Host snapshots of everything the arm freezes are taken here.
- **The inverted guard** (20 e), in ``ft_before_save``, after Amendment 18 note 2's record and before ``save_npz``:
  every tensor, row block and grid the arm froze must be bit-identical to its snapshot (``torch.equal``; a grid by
  ``e4p_hooks.qa_snapshot`` / ``qa_changed``). The differences go to the row's ``frozen_changed``, and a difference
  fails the row (it raises inside the fork's ``try``, as Amendment 18 note 2's guard does).
- **The logs** (20 e, report only), in the row's ``e7`` record: the arm's settings, ``requires_grad`` of each trained
  tensor at the row's start, the colour grid before fine-tuning (``qa_pre``, the restored pre-fine-tuning state) and
  the trained table against it and against the grid the row is saved with, for DC and AC and separately for the
  codewords and the kept rows (values outside, the largest distance beyond in grid steps), each grid's scale after over
  before, and the geometry quantizers' scale and zero point at the save. The labels' survival is E4p's
  ``labels_survived``.

Fine-tuned rows are named in full in the config (``c3dgs_ft_g`` and so on): the fork's loop names a row
``self.e4p.ft_name(row)``, which ``_Names`` leaves as it is.
"""

import os
import sys
from typing import Dict, Optional, Tuple

import torch

import e3r_hooks
import e4p_hooks
import e5_hooks

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QMIN, QMAX = -128, 127  # qint8
TRAINED = ("_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation", "_scaling_factor", "_xyz")
GEOMETRY_QA = ("opacity_qa", "scaling_qa", "scaling_factor_qa", "rotation_qa")
COLOUR = (("dc", "_features_dc", "features_dc_qa"), ("rest", "_features_rest", "features_rest_qa"))


def _import_e7():
    saved = list(sys.path)
    sys.path.insert(0, os.path.join(REPO, "bench", "gn"))
    sys.path.insert(0, os.path.join(REPO, "kaggle"))
    try:
        import e7_scenes
    finally:
        sys.path[:] = saved
    return e7_scenes


# ------------------------------------------------------------------------------ the own grid (20 d)
def own_qparams(part: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """torch's min/max qparams for qint8, per-tensor affine, from ``part``'s own minimum and maximum, computed as
    ``MinMaxObserver._calculate_qparams`` computes them: the range widened to include 0, scale = (max - min) / 255 and
    at least float32's eps, zero point = -128 - round(min / scale), clamped to [-128, 127]. Returns (scale [1], zero
    point [1], minimum, maximum). No quantizer module is called."""
    x = part.detach().float()
    mn, mx = torch.aminmax(x)
    mn_neg, mx_pos = torch.min(mn, torch.zeros_like(mn)), torch.max(mx, torch.zeros_like(mx))
    eps = torch.tensor([torch.finfo(torch.float32).eps], device=x.device)
    scale = torch.max((mx_pos - mn_neg) / float(QMAX - QMIN), eps)
    zero_point = torch.clamp(QMIN - torch.round(mn_neg / scale).to(torch.int), QMIN, QMAX)
    return scale, zero_point, mn, mx


def set_grid(qa, scale: torch.Tensor, zero_point: torch.Tensor, mn: torch.Tensor, mx: torch.Tensor) -> None:
    """Write a grid into a ``FakeQuantize``'s buffers (scale, zero point, its observer's minimum and maximum)."""
    with torch.no_grad():
        qa.scale.resize_(scale.shape).copy_(scale.to(qa.scale.device, qa.scale.dtype))
        qa.zero_point.resize_(zero_point.shape).copy_(zero_point.to(qa.zero_point.device, qa.zero_point.dtype))
        obs = getattr(qa, "activation_post_process", None)
        for name, v in (("min_val", mn), ("max_val", mx)):
            buf = getattr(obs, name, None)
            if isinstance(buf, torch.Tensor):
                buf.resize_(v.shape).copy_(v.to(buf.device, buf.dtype))


def own_grid(g) -> Dict:
    """Set both colour quantizers to the installed table's own grid (``own_qparams`` of ``_features_dc`` and
    ``_features_rest``: codewords and kept rows). Returns, per part, the shared grid it replaced, the own grid, and the
    own step over the shared step (g.7)."""
    shared = e3r_hooks._qa_state(g)
    out = {}
    for part, tname, qname in COLOUR:
        scale, zp, mn, mx = own_qparams(getattr(g, tname))
        set_grid(getattr(g, qname), scale, zp, mn, mx)
        s = float(scale.reshape(-1)[0])
        out[part] = {"shared_scale": shared[f"{part}_scale"], "shared_zero_point": shared[f"{part}_zero_point"],
                     "scale": s, "zero_point": int(zp.reshape(-1)[0]), "min": float(mn), "max": float(mx),
                     "step_ratio": s / shared[f"{part}_scale"]}
    return out


# ------------------------------------------------------------------------------ the logs (20 e)
def grid_log(values: torch.Tensor, scale: float, zero_point: int) -> Dict:
    """``values`` against an int8 grid: how many lie outside its representable range [(-128 - zp) s, (127 - zp) s],
    and the largest distance beyond it in grid steps (0 when none)."""
    v = values.detach().double().cpu().reshape(-1)
    lo, hi = (QMIN - zero_point) * scale, (QMAX - zero_point) * scale
    beyond = torch.maximum((lo - v).clamp(min=0.0), (v - hi).clamp(min=0.0)) / scale
    return {"n_values": int(v.numel()), "n_outside_grid": int(((v < lo) | (v > hi)).sum()),
            "max_steps_beyond": float(beyond.max()) if v.numel() else 0.0}


def table_logs(g, K: int, qa: Dict) -> Dict:
    """The colour table against the grid ``qa`` (``e3r_hooks._qa_state``'s form), for the codewords (rows ``[:K]``)
    and the kept rows (``[K:]``), DC and AC apart."""
    out = {}
    for block, sl in (("codewords", slice(0, K)), ("kept", slice(K, None))):
        out[block] = {part: grid_log(getattr(g, tname).detach()[sl], qa[f"{part}_scale"], qa[f"{part}_zero_point"])
                      for part, tname, _ in COLOUR}
    return out


def geometry_qa(g) -> Dict:
    return {n: {"scale": float(getattr(g, n).scale.reshape(-1)[0]),
                "zero_point": int(getattr(g, n).zero_point.reshape(-1)[0])} for n in GEOMETRY_QA if hasattr(g, n)}


def _zero_rows(sl: slice):
    def hook(grad):
        grad = grad.clone()
        grad[sl] = 0
        return grad

    return hook


class _Names:
    """``e4p``'s functions as the fork calls them (``self.e4p``), with E7's fine-tuned row names: the config already
    gives each fine-tuned row its full name (20 d), so ``ft_name`` leaves it as it is."""

    def __init__(self, mod):
        self._mod = mod

    def __getattr__(self, name):
        return getattr(self._mod, name)

    @staticmethod
    def ft_name(row: str) -> str:
        return row


class E7Hooks(e5_hooks.E5Hooks):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.e7 = _import_e7()
        self.e4p = _Names(self.e4p)
        self._arm: Optional[Dict] = None
        if self.cfg is not None:
            own = list(self.cfg.get("own_rows") or [])
            ft = list((self.cfg.get("finetune") or {}).get("rows", []))
            bad = [r for r in own if r not in self.e7.OWN_OF] + [r for r in ft if r not in self.e7.FT_ROW_OF]
            if bad or tuple(self.fork_rows) != self.e7.FORK_ROWS:
                raise ValueError(f"not E7's rows (Amendment 20 d): colour call {list(self.fork_rows)}, own {own}, "
                                 f"fine-tuned {ft}")
            self.ogc_rows = tuple(self.fork_rows)
            self.fork_rows = self.ogc_rows + tuple(own)
            self.rep["fork"]["e7"] = {"own_rows": own, "ft_rows": ft}

    # ------------------------------------------------------------------ the rows
    def _compute_rows(self) -> None:
        rows, self.fork_rows = self.fork_rows, self.ogc_rows  # E5's loop computes the OGC row only
        try:
            super()._compute_rows()
        finally:
            self.fork_rows = rows
        for own in rows:
            if own not in self.e7.OWN_OF:
                continue
            base = self.e7.OWN_OF[own]
            rec = self.rep["fork"]["rows"].setdefault(own, {"row": own})
            if base in self.tables:
                self.tables[own] = self.tables[base]
                rec.update(status="ok", table_of=base, K=int(self.tables[base][0].shape[0]))
            else:
                rec.update(status="failed", error=f"no table for {base}")
        self.flush()

    def _install(self, g, row: str) -> None:
        super()._install(g, row)
        if row in self.e7.OWN_OF:  # the same table, saved with its own grid (20 d)
            rec = self.rep["fork"]["rows"].setdefault(row, {})
            rec["own_grid"] = own_grid(g)
            qa = e3r_hooks._qa_state(g)
            K = int(self.tables[row][0].shape[0])
            rec["own_grid"]["table_vs_grid"] = table_logs(g, K, qa)
            rec["table_range"] = self.e4q.table_range(self.tables[row][0], self.e3r.C3DGSQuantizer.from_state(qa))

    def _ft_source(self, row: str) -> str:
        return self.e7.method_of(row)

    def _row_label(self, row: str) -> str:
        m, arm = self.e7.arm_of(row)
        return f"{m}, arm {arm}"

    # ------------------------------------------------------------------ the arms (20 d) and the guard (20 e)
    def _release(self) -> None:
        for h in (self._arm or {}).get("handles", []):
            h.remove()
        self._arm = None

    def ft_before_finetune(self, g, row: str, rec: Dict) -> None:
        self._release()  # a previous row's hooks, if that row failed before its save
        method, arm = self.e7.arm_of(row)
        spec = self.e7.ARMS[arm]
        K = int(self.tables[method][0].shape[0])
        info = {"arm": arm, "method": method, "K": K, "grid": spec["grid"] or "moving", "zero_rows": spec["zero_rows"],
                "no_grad": [], "qa_pre": e3r_hooks._qa_state(g),
                "requires_grad_at_start": {n: bool(getattr(g, n).requires_grad) for n in TRAINED if hasattr(g, n)}}
        if spec["grid"] == "own":
            info["own_grid"] = own_grid(g)
        if spec["grid"]:
            g.features_dc_qa.disable_observer()
            g.features_rest_qa.disable_observer()
        info["observer_enabled"] = {q: int(getattr(g, q).observer_enabled.reshape(-1)[0]) for _, _, q in COLOUR}
        handles, snap = [], {}
        if spec["zero_rows"]:
            sl = slice(0, K) if spec["zero_rows"] == "codewords" else slice(K, None)
            label = f"[:{K}]" if spec["zero_rows"] == "codewords" else f"[{K}:]"
            for _, tname, _ in COLOUR:
                handles.append(getattr(g, tname).register_hook(_zero_rows(sl)))
                snap[f"{tname}{label}"] = (tname, sl, getattr(g, tname).detach()[sl].to("cpu", copy=True))
        for n in spec["no_grad"]:
            if hasattr(g, n):
                getattr(g, n).requires_grad_(False)
                info["no_grad"].append(n)
                snap[n] = (n, slice(None), getattr(g, n).detach().to("cpu", copy=True))
        grid = e4p_hooks.qa_snapshot(g) if spec["grid"] else None
        info["frozen"] = (["colour grids"] if grid is not None else []) + sorted(snap)
        self._arm = {"row": row, "info": info, "grid": grid, "tensors": snap, "handles": handles}
        rec["e7"] = info

    def ft_before_save(self, g, row: str, rec: Dict) -> None:
        a = self._arm
        if a is None or a["row"] != row:
            raise RuntimeError(f"E7: no arm was set for {row} before its fine-tuning")
        changed = {}
        if a["grid"] is not None:
            moved = e4p_hooks.qa_changed(a["grid"], e4p_hooks.qa_snapshot(g))
            if moved:
                changed["colour grids"] = moved
        for key, (n, sl, before) in a["tensors"].items():
            now = getattr(g, n).detach()[sl].to("cpu", copy=True)
            if not torch.equal(now, before):
                d = (now.double() - before.double()).abs() if now.shape == before.shape else None
                changed[key] = {"n_differ": int((d > 0).sum()) if d is not None else None,
                                "max_abs": float(d.max()) if d is not None and d.numel() else None,
                                "shape": [list(before.shape), list(now.shape)]}
        info = a["info"]
        qa = e3r_hooks._qa_state(g)
        info["scale_after_over_before"] = {p: qa[f"{p}_scale"] / info["qa_pre"][f"{p}_scale"] for p, _, _ in COLOUR}
        info["table_vs_grid"] = {"pre": table_logs(g, info["K"], info["qa_pre"]), "saved": table_logs(g, info["K"], qa)}
        info["geometry_qa_at_save"] = geometry_qa(g)
        rec["frozen_changed"] = changed
        self._release()
        if changed:
            raise RuntimeError(f"E7's inverted guard (Amendment 20 e): frozen values moved during fine-tuning: "
                               f"{sorted(changed)}")

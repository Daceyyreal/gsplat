"""The G0 rule of kaggle/PREREG_GN.md, applied to the E0 result rows (written before any result).

G0 passes when, on garden and bicycle, (1) the three configs ordered by the predicted error ``P``
come out in the same order as by the measured error on train views and on test views, and
(2) ``0.5 <= P / D_train <= 2`` for every config. Measured errors are on images clamped to [0, 1];
the unclamped comparison is reported but not part of the rule. G0 is judged only if every validity
check holds; otherwise the verdict is "invalid".
"""

from typing import Dict, Iterable, List, Optional, Sequence

CONFIG_ORDER = ("upstream_l1", "plain_l2", "lloyd_wopa_area")
SCENES = ("garden", "bicycle")
RATIO_RANGE = (0.5, 2.0)
SEED = 0


def ordering(
    values: Dict[str, float], configs: Sequence[str] = CONFIG_ORDER
) -> List[str]:
    """Ascending; exact ties broken by the pre-registered table order."""
    return sorted(configs, key=lambda c: (values[c], configs.index(c)))


def _pick(rows: Iterable[Dict], scene: str, config: str, seed: int) -> Optional[Dict]:
    found = [
        r
        for r in rows
        if r.get("scene") == scene
        and r.get("config") == config
        and int(float(r.get("seed", -1))) == seed
    ]
    return found[-1] if found else None


def judge_scene(rows: Iterable[Dict], scene: str, seed: int = SEED) -> Dict:
    rows = list(rows)
    picked = {c: _pick(rows, scene, c, seed) for c in CONFIG_ORDER}
    missing = [c for c, r in picked.items() if r is None]
    if missing:
        return {"scene": scene, "complete": False, "missing": missing, "pass": False}
    out = {"scene": scene, "complete": True, "missing": []}
    for kind in ("clamped", "raw"):
        pred = {c: float(picked[c]["predicted"]) for c in CONFIG_ORDER}
        train = {c: float(picked[c][f"measured_train_{kind}"]) for c in CONFIG_ORDER}
        test = {c: float(picked[c][f"measured_test_{kind}"]) for c in CONFIG_ORDER}
        ratio_train = {
            c: pred[c] / train[c] if train[c] > 0 else float("inf")
            for c in CONFIG_ORDER
        }
        ratio_test = {
            c: pred[c] / test[c] if test[c] > 0 else float("inf") for c in CONFIG_ORDER
        }
        order_pred, order_train, order_test = (
            ordering(pred),
            ordering(train),
            ordering(test),
        )
        lo, hi = RATIO_RANGE
        entry = {
            "order_predicted": order_pred,
            "order_measured_train": order_train,
            "order_measured_test": order_test,
            "same_order_train": order_pred == order_train,
            "same_order_test": order_pred == order_test,
            "ratio_train": ratio_train,
            "ratio_test": ratio_test,
            "ratios_train_in_range": all(lo <= r <= hi for r in ratio_train.values()),
        }
        entry["pass"] = bool(
            entry["same_order_train"]
            and entry["same_order_test"]
            and entry["ratios_train_in_range"]
        )
        out[kind] = entry
    out["pass"] = out["clamped"]["pass"]  # the rule uses the clamped (eval) images
    return out


def judge_g0(
    rows: Iterable[Dict], validity: Dict[str, bool], scenes: Sequence[str] = SCENES
) -> Dict:
    """``validity``: name -> bool for every validity check (SH basis, toy exactness, render parity
    and reproduction per scene)."""
    rows = list(rows)
    per_scene = {s: judge_scene(rows, s) for s in scenes}
    valid = all(validity.values()) and len(validity) > 0
    complete = all(v["complete"] for v in per_scene.values())
    if not complete:
        verdict = "incomplete"
    elif not valid:
        verdict = "invalid"
    else:
        verdict = "pass" if all(v["pass"] for v in per_scene.values()) else "fail"
    return {
        "rule": (
            "on garden and bicycle: configs ordered by predicted P equal the order by measured D on train "
            "views and on test views (clamped renders), and 0.5 <= P / D_train <= 2 for all three configs"
        ),
        "configs": list(CONFIG_ORDER),
        "seed": SEED,
        "validity": validity,
        "valid": valid,
        "complete": complete,
        "verdict": verdict,
        "per_scene": per_scene,
    }

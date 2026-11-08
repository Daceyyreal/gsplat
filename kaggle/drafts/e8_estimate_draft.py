"""DRAFT, not part of any amendment: E8's runtime and quota estimate (kaggle/drafts/E8_AMENDMENT21_DRAFT.md).

    .venv/Scripts/python.exe kaggle/drafts/e8_estimate_draft.py

E7's model (``bench/gn/e7_estimate.py``, by import, unchanged; itself E6's and E5's) with E8's rows:

- **A full process** (``j-1_p0``, ``j0_p0``, ``j+1_p0``, ``j0_p1``, as E7) runs the unfine-tuned pair and arm R (E6's base
  process) plus arms G, N, C and K for both methods: 8 more fine-tuned rows at ``e6_estimate.extra_ft_row_s``; ``j0_p0``
  adds the replicate pair (2 rows). No ``*_own`` rows (E7 had 2 per full process). ``j-1_p1`` and ``j+1_p1``: arm R only.
- **The split** is E7's (20 j): every scene splits into its seed-0 half and its seed-1 half.
- **Part (c), optional:** K = 512 (OGC's equal-size row) with E6 part (b)'s four processes per scene (the unfine-tuned
  pair and arm R; j = 0 at two seeds), its own harness; VQ time taken as at K = 4,096 (an upper bound).
- **Calibration:** E7's measured halves (``kaggle/gn_e7/S*/gn7/timings_S*.json``) over E7's own high estimates, per scene
  and half, applied to E8's high estimates ("calibrated").
- **Setup on the cu130 image:** with a restored wheel 420 s (Note 4's figure; E7 S1-S10 installed in 125.9-172.4 s);
  without one, S11's 4,464.0 s (preflight 48.5 + restore 22.2 + install 4,166.9 with the 4,034.8 s wheel build + C3DGS
  build 226.7). Kaggle's 12 h is 43,200 s; Note 4 keeps 3,600 s, so the budget from setup's end is
  43,200 - setup - 3,600.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (os.path.join(REPO, "bench", "gn"), os.path.join(REPO, "kaggle")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import e6_estimate as e6e  # noqa: E402
import e7_estimate as e7e  # noqa: E402
import e7_scenes as es  # noqa: E402

E8_ARM_FT_ROWS = 8  # arms G, N, C, K x 2 methods; R is in E6's base process
E7_MEASURED = {  # timings_S#.json: gn_e7_<scene>_h0_s, gn_e7_<scene>_h1_s
    "room": (26415.5, 13020.2), "kitchen": (32205.6, 14205.3), "drjohnson": (26010.5, 13440.3),
    "bonsai": (23250.4, 10380.2), "counter": (26175.5, 11175.2), "playroom": (22755.5, 10095.2),
    "truck": (16305.3, 8430.2)}
SETUP_RESTORED = 420.0
SETUP_WHEEL_BUILD = 48.5 + 22.2 + 4166.9 + 226.7
KAGGLE_S, SPARE_S = 43200.0, 3600.0
WEEKLY_QUOTA_H = 30.0  # assumption: Kaggle's weekly GPU quota


def parts(scene, end, U, ft_rows=E8_ARM_FT_ROWS):
    pre, e6p = e6e.scene_parts(scene, end, U)
    ft = e6e.extra_ft_row_s(scene, end, U)
    out = []
    for p in es.PROCESSES:
        s = e6p[e7e.E6_INDEX[(p["j"], p["seed"])]]
        if p["full"]:
            s += ft_rows * ft + (e7e.REPLICATE_FT_ROWS * ft if p["replicate"] else 0.0)
        out.append(s)
    return pre, out


def half(scene, h, end, U):
    pre, procs = parts(scene, end, U)
    return (pre + sum(procs[i] for i, _ in es.half_processes(scene, (h,)))) * e7e.cpu_x(scene)


def part_c(scene, end, U):
    pre, e6p = e6e.scene_parts(scene, end, U)
    return (pre + sum(e6p)) * e7e.cpu_x(scene)


def ratio(scene, h, U):
    return E7_MEASURED[scene][h] / e7e.half_s(scene, h, 1, U)


# one lane per session (as E7): the seven seed-0 halves alone, then the seed-1 halves two to a session where both fit
PLAN = [[("room", 0)], [("kitchen", 0)], [("drjohnson", 0)], [("bonsai", 0)], [("counter", 0)], [("playroom", 0)],
        [("truck", 0)], [("kitchen", 1), ("truck", 1)], [("room", 1), ("playroom", 1)], [("bonsai", 1), ("counter", 1)],
        [("drjohnson", 1)]]


if __name__ == "__main__":
    U = e7e.units()
    T = {"lo": 0.0, "hi": 0.0, "cal": 0.0, "c_hi": 0.0, "c_cal": 0.0}
    print("scene      seed-0 half low-high (calibrated)      seed-1 half low-high (calibrated)      part (c) high")
    for s in e7e.GATE:
        lo, hi = [half(s, h, 0, U) for h in (0, 1)], [half(s, h, 1, U) for h in (0, 1)]
        cal = [hi[h] * ratio(s, h, U) for h in (0, 1)]
        c = part_c(s, 1, U)
        cc = c * sum(E7_MEASURED[s]) / sum(e7e.half_s(s, h, 1, U) for h in (0, 1))
        T["lo"] += sum(lo); T["hi"] += sum(hi); T["cal"] += sum(cal); T["c_hi"] += c; T["c_cal"] += cc
        print(f"{s:9}  {lo[0]:7,.0f}-{hi[0]:7,.0f} ({cal[0]:7,.0f})   {lo[1]:7,.0f}-{hi[1]:7,.0f} ({cal[1]:7,.0f})   "
              f"{c:7,.0f}")
    print({k: f"{v:,.0f} s = {v / 3600:.1f} h" for k, v in T.items()})
    budget = {n: KAGGLE_S - x - SPARE_S for n, x in (("restored wheel", SETUP_RESTORED),
                                                     ("wheel built in the session", SETUP_WHEEL_BUILD))}
    print("budget from setup's end:", {k: f"{v:,.0f} s" for k, v in budget.items()})
    tot_hi = tot_cal = 0.0
    for i, lane in enumerate(PLAN, 1):
        hi = sum(half(s, h, 1, U) for s, h in lane)
        cal = sum(half(s, h, 1, U) * ratio(s, h, U) for s, h in lane)
        tot_hi += hi + SETUP_RESTORED
        tot_cal += cal + SETUP_RESTORED
        print(f"S{i:<2} {', '.join(f'{s} h{h}' for s, h in lane):30} high {hi:7,.0f} s, spare "
              f"{budget['restored wheel'] - hi:7,.0f} (restored) / {budget['wheel built in the session'] - hi:7,.0f} "
              f"(built); session-hours high {(hi + SETUP_RESTORED) / 3600:5.2f}, calibrated {(cal + SETUP_RESTORED) / 3600:5.2f}")
    wheel = SETUP_WHEEL_BUILD / 3600
    print(f"E8 (a)+(b): {len(PLAN)} sessions + one wheel-build session ({wheel:.2f} h): "
          f"{tot_hi / 3600 + wheel:.1f} session-hours high, {tot_cal / 3600 + wheel:.1f} calibrated; "
          f"at {WEEKLY_QUOTA_H:.0f} h a week: {(tot_hi / 3600 + wheel) / WEEKLY_QUOTA_H:.1f} / "
          f"{(tot_cal / 3600 + wheel) / WEEKLY_QUOTA_H:.1f} weeks")

# ------------------------------------------------------------------ the 2x2 redesign (2026-11-08, second session)
# Arms per full process, beyond E6's base process (unfine-tuned pair + one fine-tuned pair): the base pair becomes G
# (anchor), so the 4-arm design (Z = the unfine-tuned rows, C, N, G) adds C and N (4 rows) and no replicate; the 5-arm
# design adds O (opacity only, 2 more rows); the first draft (R, G, N, C, K, R2) added 8 rows and the replicate pair.
# j-1_p1 / j+1_p1 run the anchor's pair (G, or R in the first draft): the same cost as E6's base process.
VARIANTS = {"4-arm (Z, C, N, G)": (4, False), "5-arm (Z, C, N, O, G)": (6, False),
            "6-arm first draft (R, G, N, C, K, R2)": (8, True)}


def half_v(scene, h, end, U, extra, replicate):
    pre, e6p = e6e.scene_parts(scene, end, U)
    ft = e6e.extra_ft_row_s(scene, end, U)
    procs = []
    for p in es.PROCESSES:
        s = e6p[e7e.E6_INDEX[(p["j"], p["seed"])]]
        if p["full"]:
            s += extra * ft + (e7e.REPLICATE_FT_ROWS * ft if (replicate and p["replicate"]) else 0.0)
        procs.append(s)
    return (pre + sum(procs[i] for i, _ in es.half_processes(scene, (h,)))) * e7e.cpu_x(scene)


def variant(name, U, budget=KAGGLE_S - SETUP_RESTORED - SPARE_S):
    """Jobs (high, calibrated), the sessions (seed-0 halves one each; seed-1 halves first-fit decreasing into the
    budget), session-hours with a 420 s setup each plus one wheel session, and weeks at WEEKLY_QUOTA_H."""
    extra, rep = VARIANTS[name]
    halves = [(half_v(s, h, 1, U, extra, rep), s, h) for s in e7e.GATE for h in (0, 1)]
    hi = sum(x for x, _, _ in halves)
    cal = sum(x * ratio(s, h, U) for x, s, h in halves)
    bins = []
    for x in sorted((x for x, _, h in halves if h == 1), reverse=True):
        for b in bins:
            if sum(b) + x <= budget:
                b.append(x)
                break
        else:
            bins.append([x])
    n = sum(1 for _, _, h in halves if h == 0) + len(bins)
    wheel = SETUP_WHEEL_BUILD / 3600
    sh = (hi + n * SETUP_RESTORED) / 3600 + wheel, (cal + n * SETUP_RESTORED) / 3600 + wheel
    return {"jobs_h": (hi / 3600, cal / 3600), "sessions": n, "session_h": sh,
            "weeks": (sh[0] / WEEKLY_QUOTA_H, sh[1] / WEEKLY_QUOTA_H),
            "largest_half_s": max(x for x, _, _ in halves)}


if __name__ == "__main__":
    U = e7e.units()
    print()
    for name in VARIANTS:
        v = variant(name, U)
        print(f"{name}: jobs {v['jobs_h'][0]:.1f} h high / {v['jobs_h'][1]:.1f} h calibrated; {v['sessions']} sessions + wheel; "
              f"{v['session_h'][0]:.1f} / {v['session_h'][1]:.1f} session-hours; {v['weeks'][0]:.1f} / {v['weeks'][1]:.1f} weeks; "
              f"largest half {v['largest_half_s']:,.0f} s")

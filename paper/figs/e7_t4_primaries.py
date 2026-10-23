"""E7 T4: the primaries and their tests (kaggle/PREREG_GN.md Amendment 20 f, h), as bench/gn/e7.py computes them.

    .venv/Scripts/python.exe paper/figs/e7_t4_primaries.py [--root kaggle/gn_e7] [--out paper/figs/out]

From ``e7.verdict(root)`` only (no BD, noise or test is recomputed here):

- ``e7_t4_primaries.tex``: per scene BD_R, BD_G and BD_CG (seed-0 curves; "--" for a BD not computed, which enters the
  primaries as 0), Delta_refit = BD_G - BD_R and Delta_cb = BD_CG - BD_G; footer rows Delta_bar, SD_pool and
  SE = SD_pool / sqrt(n) per primary.
- ``e7_t4_tests.tex``: per primary and test type (superiority, equivalence at delta = 0.07 dB), the Holm position, the
  statistic (|Delta_bar| / SE, or the TOST margin m), the critical t, the result; then the outcome of 20 h, and the
  robustness reading's outcome under SE_BD and the fragile flag; an incomplete primary with its reasons.
- ``e7_t4_primaries.csv``: every value at full precision.
"""

import e7_figdata as ed

SCRIPT = "e7_t4_primaries"


def main(root, out):
    v = ed.verdict(root)
    P = v["primaries"]
    rows, body = [], []
    for s in v["scenes"]:
        r, c = P["refit"]["per_scene"].get(s, {}), P["cb"]["per_scene"].get(s, {})
        vals = [r.get("BD_R"), r.get("BD_G"), c.get("BD_CG"), r.get("delta"), c.get("delta")]
        rows.append(["scene", s] + vals)
        body.append([s] + [ed.num(x, 6, sign=True) for x in vals])
    foot = []
    for lab, f in (("$\\bar\\Delta$", lambda k: P[k]["delta_bar"]), ("SD_pool", lambda k: P[k]["noise"]["SD_pool"]),
                   ("SE = SD_pool / $\\sqrt{n}$", lambda k: P[k]["noise"]["SE"])):
        a, b = f("refit"), f("cb")
        rows.append(["stat", lab, None, None, None, a, b])
        foot.append([lab, "", "", "", ed.num(a, 6, sign=lab.startswith("$")), ed.num(b, 6, sign=lab.startswith("$"))])
    ed.write_tex(out, "e7_t4_primaries.tex", SCRIPT, "Amendment 20 f: BD per arm and the two primaries (bench/gn/e7.py)",
                 "lrrrrr", ["Scene", "BD_R", "BD_G", "BD_CG", r"$\Delta_{\mathrm{refit}}$", r"$\Delta_{\mathrm{cb}}$"],
                 body, foot)

    T, rob = v["tests"], v["robustness"]
    tbody = []
    for k in ("refit", "cb"):
        sup, eqv = T["superiority"]["results"].get(k), T["equivalence"]["results"].get(k)
        if sup is None or eqv is None:
            why = "; ".join(v["incomplete_reasons"][k]) or "not tested"
            tbody.append([k, "superiority", "--", "--", "--", "incomplete"])
            tbody.append([k, "equivalence", "--", "--", "--", "incomplete"])
            rows.append(["test", k, "incomplete", why, None, None, None])
        else:
            tbody.append([k, "superiority", str(sup["position"]), ed.num(sup["statistic"], 3), ed.num(sup["critical_t"], 3),
                          ("superior, sign " + ("+" if sup["sign"] > 0 else "-")) if sup["superior"] else
                          ("not superior" if sup["tested"] else "not tested (Holm)")])
            tbody.append([k, "equivalence", str(eqv["position"]), ed.num(eqv["m"], 3), ed.num(eqv["critical_t"], 3),
                          "equivalent" if eqv["equivalent"] else ("not equivalent" if eqv["tested"] else
                                                                  "not tested (Holm)")])
            rows.append(["test", k, "superiority", sup["position"], sup["statistic"], sup["critical_t"], sup["superior"]])
            rows.append(["test", k, "equivalence", eqv["position"], eqv["m"], eqv["critical_t"], eqv["equivalent"]])
        tbody.append([k, "outcome (SE)", "", "", "", v["outcomes"][k]])
        tbody.append([k, "outcome (SE_BD)", "", "", "", rob["outcomes"][k] + ("; fragile" if v["fragile"][k] else "")])
        rows.append(["outcome", k, v["outcomes"][k], rob["outcomes"][k], v["fragile"][k], None, None])
    rows.append(["order", "superiority", " > ".join(T["superiority"]["order"]), None, None, None, None])
    rows.append(["order", "equivalence", " > ".join(T["equivalence"]["order"]), None, None, None, None])
    rows.append(["robustness", "SE_BD", rob["SE_BD"], rob["SD_BD"]["SD_BD"], rob["SD_BD"]["n_used"], None, None])
    ed.write_tex(out, "e7_t4_tests.tex", SCRIPT, "Amendment 20 f, h: Holm-ordered tests, outcomes, robustness "
                 f"(delta = {v['delta']} dB; n = {v['n']})", "llrrrl",
                 ["Primary", "Test", "Holm position", "Statistic", "Critical t", "Result"], tbody)
    ed.write_csv(out, "e7_t4_primaries.csv", ["kind", "key", "a", "b", "c", "d", "e"], rows)


if __name__ == "__main__":
    ed.cli(main, __doc__)

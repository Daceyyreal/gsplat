"""E7 T5: the secondaries (kaggle/PREREG_GN.md Amendment 20 g; report only, no bars), as bench/gn/e7.py computes them.

    .venv/Scripts/python.exe paper/figs/e7_t5_secondaries.py [--root kaggle/gn_e7] [--out paper/figs/out]

From ``e7.verdict(root)["secondaries"]`` only:

- ``e7_t5_secondaries.tex``: per scene g.1 BD_K - BD_CG and BD_K - BD_R, g.2 BD_O - BD_G, g.3 P1_own - P1, g.5 P_FT
  (BD_R) and P1 from the seed-0 and the seed-1 curves, g.6 the replicate pair's difference at j = 0; footer rows the
  means with SE = SD_pool / sqrt(n) for g.1-g.2, SD_BD for g.5 (with the scenes used) and the across-scene SD for g.6.
- ``e7_t5_secondaries.csv``: every value at full precision.
"""

import e7_figdata as ed

SCRIPT = "e7_t5_secondaries"


def main(root, out):
    v = ed.verdict(root)
    g = v["secondaries"]
    parts = {"K-CG": g["g1_codebook_alone"]["K_minus_CG"], "K-R": g["g1_codebook_alone"]["K_minus_R"],
             "O-G": g["g2_own_grid_arm"]["O_minus_G"]}
    own = g["g3_P1_own_grid"]["P1_own_minus_P1"]
    g5 = g["g5_bd_noise"]
    rep = g["g6_replicate"]
    rows, body = [], []
    for s in v["scenes"]:
        vals = [parts[k]["per_scene"].get(s, {}).get("delta") for k in parts] + [own.get(s)]
        vals += [g5["P_FT"]["per_scene"][s]["seed0"], g5["P_FT"]["per_scene"][s]["seed1"],
                 g5["P1"]["per_scene"][s]["seed0"], g5["P1"]["per_scene"][s]["seed1"],
                 rep["per_scene"][s]["pair_replicate_minus_R"]]
        rows.append([s] + vals)
        body.append([s] + [ed.num(x, 4, sign=True) for x in vals])
    mean = [parts[k]["delta_bar"] for k in parts]
    se = [parts[k]["noise"]["SE"] for k in parts]
    own_v = [x for x in own.values() if x is not None]
    own_mean = sum(own_v) / len(own_v) if own_v else None
    foot = [["mean"] + [ed.num(x, 4, sign=True) for x in mean] + [ed.num(own_mean, 4, sign=True)] + [""] * 5,
            ["SE = SD_pool / $\\sqrt{n}$"] + [ed.num(x, 4) for x in se] + [""] * 6,
            ["SD_BD (scenes used)", "", "", "", "",
             f"{ed.num(g5['P_FT']['SD_BD'], 4)} ({g5['P_FT']['n_used']})", "",
             f"{ed.num(g5['P1']['SD_BD'], 4)} ({g5['P1']['n_used']})", "", ""],
            ["SD across scenes", "", "", "", "", "", "", "", "",
             ed.num(rep["sd_across_scenes"]["pair_replicate_minus_R"], 4)]]
    rows += [["stat:mean"] + mean + [own_mean] + [None] * 5, ["stat:SE"] + se + [None] * 6,
             ["stat:SD_BD", None, None, None, None, g5["P_FT"]["SD_BD"], g5["P_FT"]["n_used"], g5["P1"]["SD_BD"],
              g5["P1"]["n_used"], None],
             ["stat:replicate_sd", None, None, None, None, None, None, None, None,
              rep["sd_across_scenes"]["pair_replicate_minus_R"]]]
    ed.write_tex(out, "e7_t5_secondaries.tex", SCRIPT, "Amendment 20 g: secondaries, report only (bench/gn/e7.py)",
                 "lrrrrrrrrr", ["Scene", "K - CG", "K - R", "O - G", "P1_own - P1", "P_FT s0", "P_FT s1", "P1 s0",
                                "P1 s1", "Replicate"], body, foot)
    ed.write_csv(out, "e7_t5_secondaries.csv", ["scene", "K_minus_CG", "K_minus_R", "O_minus_G", "P1_own_minus_P1",
                                                 "P_FT_seed0", "P_FT_seed1", "P1_seed0", "P1_seed1",
                                                 "replicate_pair_minus_R"], rows)


if __name__ == "__main__":
    ed.cli(main, __doc__)

"""A stand-in for C3DGS's ``npz2ply.py``: the decoded model (the colour table dequantized and indexed, the geometry
table indexed) in INRIA's ``.ply`` layout, through ``e3p_inria.write_inria_ply``."""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ["E3R_FAKE_KAGGLE"])
import e3p_inria as ei  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("npz_file")
p.add_argument("--ply_file", required=True)
a = p.parse_args()
z = np.load(a.npz_file)


def deq(name):
    return (torch.from_numpy(z[name]).float() - float(z[f"{name}_zero_point"])) * float(z[f"{name}_scale"])


fi = torch.from_numpy(z["feature_indices"]).long()
gi = torch.from_numpy(z["gaussian_indices"]).long()
ei.write_inria_ply(a.ply_file, {
    "means": torch.from_numpy(z["xyz"]).float(), "quats": torch.from_numpy(z["rotation"])[gi],
    "scales": torch.from_numpy(z["scaling"])[gi], "opacities": torch.from_numpy(z["opacity"])[:, 0],
    "sh0": deq("features_dc")[fi], "shN": deq("features_rest")[fi]})

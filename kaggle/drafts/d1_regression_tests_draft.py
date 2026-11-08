"""DRAFT (kaggle/drafts; not collected by the gate, nothing run on Kaggle): the regression tests D1's fork path must pass
before any D1 session (R13, 2026-11-08, fourth session). They move into ``bench/gn/test_gn.py`` with D1's code, after
D1's declaration is committed alone.

R9 (``test_gn.py::test_c3dgs_color_compress_non_dir_false_loses_the_dc``) shows C3DGS's ``color_compress_non_dir=False``
branch at 2a234af5 decoding 15 coefficients with the first AC coefficient as the DC. Two contracts follow:

1. **D1's own rows** (``c3dgs``, ``ogc_gram``, ``ogc_gram_ac``) keep C3DGS's default container: the DC is a coefficient of
   every codeword, ``color_compress_non_dir`` is True, the decoded model has 16 coefficients and kept splats decode
   exactly. ``ogc_gram_ac`` changes the metric only (``ac_only_metric``), never the container.
2. **Any AC-only container** (an AC codebook with a per-splat DC: a MesonGS-like second host, or a later D1 variant) must
   decode the true per-splat DC and 15 AC coefficients from the AC codebook, i.e. 16 coefficients in all. C3DGS's
   unfixed branch fails this contract; the reference decode below (a sketch of the fork change) passes it.

Run by hand against C3DGS's source: ``GN_C3DGS_SRC=F:/c3dgs_src_2a234af5 .venv/Scripts/python.exe -m pytest
kaggle/drafts/d1_regression_tests_draft.py -q -p no:cacheprovider`` (from the repository root, ``bench/gn`` on the path).
"""
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "bench", "gn"))
from test_gn import _c3dgs_modules  # noqa: E402  (the same CPU loader as R9's repro)

N, K = 64, 8


def _src():
    src = os.environ.get("GN_C3DGS_SRC")
    if not src or not os.path.isfile(os.path.join(src, "compression", "vq.py")):
        pytest.skip("needs C3DGS's source at 2a234af5 in GN_C3DGS_SRC")
    return src


def _setup(monkeypatch):
    import torch

    gm, vq = _c3dgs_modules(monkeypatch, _src())
    torch.manual_seed(0)
    feats, importance = torch.randn(N, 16, 3), torch.rand(N)

    def stub_vq(features, importance, codebook_size, *a, **kw):  # the first k rows as codebook, round-robin indices
        return features[:codebook_size].clone(), torch.arange(len(features)) % codebook_size

    monkeypatch.setattr(vq, "vq_features", stub_vq)
    settings = vq.CompressionSettings(codebook_size=K, importance_prune=None, importance_include=0.5, steps=1,
                                      decay=0.8, batch_size=16)
    return gm, vq, feats, importance, settings


def _model(gm, feats, set_color_indexed=None):
    g = types.SimpleNamespace(_feature_indices=None, color_index_mode=None, _features_dc=None, _features_rest=None,
                              features_dc_qa=lambda x: x, features_rest_qa=lambda x: x)
    g.get_features = feats  # what compress_color reads
    g.set_color_indexed = set_color_indexed or (lambda f, i: gm.GaussianModel.set_color_indexed(g, f, i))
    return g


def assert_ac_only_contract(decoded, feats, kept):
    """The AC-only container's contract: 16 coefficients, the true per-splat DC on every splat, kept splats exact."""
    import torch

    assert decoded.shape == feats.shape, f"decoded {tuple(decoded.shape)}, expected {tuple(feats.shape)}"
    assert torch.equal(decoded[:, 0], feats[:, 0]), "the decoded DC is not the true DC"
    assert torch.equal(decoded[kept], feats[kept]), "a kept splat does not decode exactly"


def test_d1_rows_use_the_default_container(monkeypatch):
    """Contract 1: the call D1's hook makes (``color_compress_non_dir=True``) decodes 16 coefficients, kept rows exact;
    the hook refuses ``False``."""
    import torch

    gm, vq, feats, importance, settings = _setup(monkeypatch)
    calls = []
    real = vq.compress_color

    def guarded(g, imp, comp, non_dir):  # what D1's hook (kaggle/d1_hooks.py) is to wrap compress_color with
        calls.append(non_dir)
        if non_dir is not True:
            raise AssertionError("D1 never takes C3DGS's color_compress_non_dir=False branch")
        return real(g, imp, comp, non_dir)

    g = _model(gm, feats)
    guarded(g, importance, settings, True)
    full = gm.GaussianModel.get_features.fget(g)
    kept = importance > 0.5
    assert calls == [True] and full.shape == (N, 16, 3) and torch.equal(full[kept], feats[kept])
    with pytest.raises(AssertionError, match="never takes"):
        guarded(_model(gm, feats), importance, settings, False)


def test_unfixed_c3dgs_branch_fails_the_ac_only_contract(monkeypatch):
    """C3DGS's branch at 2a234af5, unfixed, fails contract 2 (R9's finding, as a failing contract)."""
    gm, vq, feats, importance, settings = _setup(monkeypatch)
    g = _model(gm, feats)
    vq.compress_color(g, importance, settings, False)
    with pytest.raises(AssertionError, match="decoded"):
        assert_ac_only_contract(gm.GaussianModel.get_features.fget(g), feats, importance > 0.5)


def test_reference_ac_only_decode_passes_the_contract(monkeypatch):
    """A sketch of the fork change (``scene/gaussian_model.py``: a per-splat DC beside the indexed AC table;
    ``compression/vq.py``: the AC-only branch hands the DC over) passes contract 2."""
    import torch

    gm, vq, feats, importance, settings = _setup(monkeypatch)
    dc = feats[:, :1].clone()
    store = {}

    def set_color_indexed_ac_only(ac_table, indices):  # ac_table: (rows, 15, 3), indices: one row per splat
        assert ac_table.shape[1:] == (15, 3)
        store.update(dc=dc, ac=ac_table, idx=indices)

    g = _model(gm, feats, set_color_indexed_ac_only)
    vq.compress_color(g, importance, settings, False)
    decoded = torch.cat((store["dc"], store["ac"][store["idx"]]), dim=1)  # the fork's get_features in its AC mode
    assert_ac_only_contract(decoded, feats, importance > 0.5)
    assert not torch.equal(decoded[~(importance > 0.5)][:, 1:], feats[~(importance > 0.5)][:, 1:])  # AC was quantized

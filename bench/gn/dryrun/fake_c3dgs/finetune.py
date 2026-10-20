"""A stand-in for C3DGS's ``finetune.py``: the same signature; ``training_setup`` (an optimizer appears on the model);
cameras drawn with Python's ``random.randint`` as C3DGS's loop draws them; three small steps on the colour table from
torch's global stream (the indices untouched), standing in for its 5,000 iterations.

With ``E7_FAKE_FINETUNE=1`` (E7, PREREG_GN.md Amendment 20; off by default) it trains as C3DGS's ``finetune`` does
instead: five steps of a loss through the model's fake-quantized colours (``get_features``: the observers update when
they are on, and a value outside the grid gets no gradient), opacity and geometry tables, with ``loss.backward()``
and C3DGS's Adam (``training_setup``), stepping and zeroing the gradients as its loop does."""

import os
import random

import torch


def _finetune_grad(scene, g):
    stack = None
    for _ in range(5):
        if not stack:
            stack = list(scene.getTrainCameras())
        stack.pop(random.randint(0, len(stack) - 1))
        f = g.get_features
        loss = ((f - 0.1) ** 2).mean() + 1e-2 * g.opacity_qa(torch.sigmoid(g._opacity)).mean()
        loss = loss + 1e-2 * (g.scaling_qa(torch.exp(g._scaling)).mean() + g.scaling_factor_qa(g._scaling_factor).mean()
                              + g.rotation_qa(g._rotation).pow(2).mean()) + 1e-3 * g._xyz.pow(2).mean()
        loss.backward()
        g.optimizer.step()
        g.optimizer.zero_grad()


def finetune(scene, dataset, opt, comp, pipe, testing_iterations, debug_from):
    g = scene.gaussians
    g.training_setup(opt)
    if os.environ.get("E7_FAKE_FINETUNE") == "1":
        return _finetune_grad(scene, g)
    stack = None
    with torch.no_grad():
        for _ in range(3):
            if not stack:
                stack = list(scene.getTrainCameras())
            stack.pop(random.randint(0, len(stack) - 1))
            g._features_rest += torch.randn_like(g._features_rest) * 1e-3
            g._features_dc += torch.randn_like(g._features_dc) * 1e-3

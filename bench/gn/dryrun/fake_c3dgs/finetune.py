"""A stand-in for C3DGS's ``finetune.py``: the same signature; ``training_setup`` (an optimizer appears on the model);
cameras drawn with Python's ``random.randint`` as C3DGS's loop draws them; three small steps on the colour table from
torch's global stream (the indices untouched), standing in for its 5,000 iterations."""

import random

import torch


def finetune(scene, dataset, opt, comp, pipe, testing_iterations, debug_from):
    g = scene.gaussians
    g.training_setup(opt)
    stack = None
    with torch.no_grad():
        for _ in range(3):
            if not stack:
                stack = list(scene.getTrainCameras())
            stack.pop(random.randint(0, len(stack) - 1))
            g._features_rest += torch.randn_like(g._features_rest) * 1e-3
            g._features_dc += torch.randn_like(g._features_dc) * 1e-3

"""Emulated devices for the stand-in's evaluation (E5p, PREREG_GN.md Amendment 17 e.2), on a machine with no GPU.

C3DGS renders on CUDA and loads the ground-truth images on ``--data_device``; its ``render_and_eval`` passes the image
as loaded, so with ``--data_device cpu`` its ``ssim`` raises. A CPU-only dry run cannot reach that state with real
tensors, so with ``E3R_FAKE_DEVICES=1`` the stand-in's renders and images are ``Emu`` tensors: real CPU tensors whose
``device`` reports the emulated one ("cuda" for a render, ``--data_device`` for an image), carried through torch ops,
and whose ``.to(device)`` moves only the tag. The stand-in's ``ssim``, ``psnr`` and ``lpips`` raise C3DGS's own error
when the two devices differ. Without the variable everything is a plain tensor (E3r-E4q's stand-in, unchanged)."""

import os

import torch

ON = os.environ.get("E3R_FAKE_DEVICES") == "1"
RENDER_DEVICE = "cuda"
MISMATCH = ("Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same or input should "
            "be a MKLDNN tensor and weight is a dense tensor")


class Emu(torch.Tensor):
    @staticmethod
    def __new__(cls, t: torch.Tensor, device: str):
        r = torch.Tensor._make_subclass(cls, t.detach())
        r._emu = str(device)
        return r

    @property
    def device(self):
        return torch.device(self._emu)

    def to(self, *a, **k):
        dev = k.get("device", a[0] if a else None)
        if isinstance(dev, (str, torch.device)):
            return Emu(plain(self), torch.device(dev).type)
        return super().to(*a, **k)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        tag = next((x._emu for x in _flat(args) + _flat(list(kwargs.values())) if isinstance(x, Emu)), RENDER_DEVICE)
        with torch._C.DisableTorchFunctionSubclass():
            out = func(*args, **kwargs)
        return _wrap(out, tag)


def _flat(xs):
    out = []
    for x in xs:
        out.extend(_flat(x) if isinstance(x, (list, tuple)) else [x])
    return out


def _wrap(out, tag):
    if isinstance(out, torch.Tensor) and not isinstance(out, Emu):
        return Emu(out, tag)
    if isinstance(out, (list, tuple)):
        return type(out)(_wrap(o, tag) for o in out)
    return out


def plain(t):
    """The underlying CPU tensor."""
    if isinstance(t, Emu):
        with torch._C.DisableTorchFunctionSubclass():
            return t.as_subclass(torch.Tensor)
    return t


def on(t: torch.Tensor, device: str):
    """``t`` on the emulated ``device`` when the emulation is on, else ``t`` itself."""
    return Emu(t, device) if ON else t


def check_same_device(a, b) -> None:
    if ON and getattr(a, "device", None) != getattr(b, "device", None):
        raise RuntimeError(MISMATCH)

import torch
from fastcore.all import *
from fastai.torch_core import *
from .presentation import *

class TensorAudioBase(TensorBase):
    def __new__(cls, x, sample_rate=32000, cache_path=None, orig_path=None, **kwargs):
        x = tensor(x)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim == 2:
            pass
        else:
            pass
            # raise ValueError(f"TensorAudio expects input with shape (samples,) or (channels, samples), got {x.shape}")

        res = super().__new__(cls, x, **kwargs)
        res.sample_rate = sample_rate
        res.cache_path = cache_path
        res.orig_path = orig_path
        return res

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        res = super().__torch_function__(func, types, args, kwargs)
        if isinstance(res, torch.Tensor):
            res = retain_type(res, self)
            res.sample_rate = self.sample_rate
        return res

    _show_args = {} # TODO - (see TensorImage)
    def show(self, ctx=None, **kwargs):
        return show_audio(self, ctx=ctx, **{**self._show_args, **kwargs})

class TensorAudio(TensorAudioBase):
    pass
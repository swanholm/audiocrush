from fastai.data.core import show_batch
from fastai.data.transforms import Category
from fastai.vision.data import get_grid
import torch
from fastcore.all import *
from fastai.torch_core import *
from .presentation import *

class TensorAudio(TensorBase):
    def __new__(cls, x, sample_rate=32000, cache_path=None, orig_path=None, **kwargs):
        x = tensor(x)

        res = super().__new__(cls, x, **kwargs)
        res.sample_rate = sample_rate
        res.cache_path = cache_path
        res.orig_path = orig_path
        return res
    
    # def __torch_function__(self, func, types, args=(), kwargs=None):
    #     if kwargs is None:
    #         kwargs = {}
    #     res = super().__torch_function__(func, types, args, kwargs)
    #     if isinstance(res, torch.Tensor):
    #         res = retain_type(res, self)
    #         res.sample_rate = self.sample_rate
    #         res.cache_path = self.cache_path
    #         res.orig_path = self.orig_path
    #     return res
    
    # def __repr__(self):
    #     return f"TensorAudio(shape={tuple(self.shape)}, sr={self.sample_rate}, path={self.orig_path})"
    
    # def __format__(self, format_spec):
    #     return torch.Tensor.__format__(self, format_spec)

    # def __getitem__(self, idx):
    #     return _getitem(self, idx)

    _show_args = {} # TODO - (see TensorImage)
    def show(self, ctx=None, **kwargs):
        return show_audio(self, ctx=ctx, **{**self._show_args, **kwargs})


# class TensorAudio(TensorAudioBase):
#     pass

# @typedispatch
# def _getitem(x:TensorAudioBase, idx:Category):
#     index = idx.items if hasattr(idx, 'items') else idx
#     res = x[index] # ??!!  check if this is correct
#     if res.ndim == 3:
#         res = res[0]
#     return res

# @typedispatch
# def _getitem(x:TensorAudioBase, idx:slice):
#     res = TensorAudioBase(super(TensorAudioBase, x).__getitem__(idx))
#     res.sample_rate = x.sample_rate
#     res.cache_path = x.cache_path
#     res.orig_path = x.orig_path
#     return res

# @typedispatch
# def _getitem(x:TensorAudioBase, idx:int):
#     if idx is None:
#         res = x
#     elif x.ndim == 2:
#         res = TensorAudioBase(super(TensorAudioBase, x).__getitem__(idx))
#         res.sample_rate = x.sample_rate
#         res.cache_path = x.cache_path
#         res.orig_path = x.orig_path
#     elif x.ndim == 3:
#         res = TensorAudioBase(super(TensorAudioBase, x).__getitem__(idx))
#         res.sample_rate = x.sample_rate
#         res.cache_path = x.cache_path
#         res.orig_path = x.orig_path
#     return res

@typedispatch
def plot_top_losses(x:TensorAudio, y, its, outs, preds, losses, **kwargs):
    k = len(its)
    fig, axes = plt.subplots(k, 1, figsize=(10, 2*k))
    axes = axes.flat if k > 1 else [axes]
    for i,it in enumerate(its):
        ax = axes[i]
        it[0].show(ctx=ax)
        title = 'todo' # f'Label: {it[1]} ({y[idx]}), Pred: {y.vocab[preds[idx]]} ({preds[idx]}), Loss: {losses[idx]:.4f}'
        ax.set_title(title)
    plt.tight_layout()
    return its

@typedispatch
def show_batch(x: TensorAudio, y, samples, ctxs=None, max_n=10, **kwargs):
    k = min(len(samples), max_n)
    if ctxs is None:
        fig, axes = plt.subplots(k, 1, figsize=(10, 2*k))
        ctxs = axes.flat if k > 1 else [axes]
    for i, (audio, label) in enumerate(samples[:k]):
        ctx = ctxs[i]
        audio.show(ctx=ctx)
        title = f'{label} [{i}]'
        ctx.set_title(title)
    plt.tight_layout()
    return ctxs

# @typedispatch
# def show_batch(
#         x: TensorAudioBase,
#         y,
#         samples,
#         ctxs=None,
#         max_n=10,
#         nrows=5,
#         ncols=2,
#         figsize=None,
#         **kwargs,
#     ):
#         if figsize is None:
#             figsize = (2, 5)

#         if ctxs is None:
#             ctxs = get_grid(
#                 min(len(samples), max_n), nrows=nrows, ncols=ncols, figsize=figsize
#             )
#         ctxs = show_batch[object](
#             x, y, samples, ctxs=ctxs, max_n=max_n, **kwargs
#         )
#         return ctxs
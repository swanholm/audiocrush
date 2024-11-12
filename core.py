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

    _show_args = {} # TODO - (see TensorImage)
    def show(self, ctx=None, **kwargs):
        return show_audio(self, ctx=ctx, **{**self._show_args, **kwargs})

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
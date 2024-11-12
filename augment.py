import subprocess
from fastai.data.transforms import Transform
from fastai.data.block import TransformBlock
from fastai.data.transforms import get_files
from pathlib import Path
from audiocrush.core import TensorAudio

from fastai.torch_basics import *

from fastai.vision.augment import RandTransform
import torchaudio
import torch


import random
import torchaudio
import torch
from fastai.vision.augment import RandTransform
from audiocrush.core import TensorAudio
import math

class RandomResample(RandTransform):
    order = 10  # order in the pipeline in which the transform is applied
    split_idx = 0
    do_decode = False

    def __init__(
        self,
        min_factor=0.8,         
        max_factor=1.25,            
        p=0.5, # probability of applying the transform
    ):
        super().__init__(p=p)
        self.min_factor = min_factor
        self.max_factor = max_factor
        
def encodes(self, x: TensorAudio):
    # perform a very crude but fast resampling using linear interpolation
    orig_length = x.shape[-1]
    scaling_factor = random.uniform(self.min_factor, self.max_factor)

    if scaling_factor > 1:
        # upsampling: compute the required input length and truncate
        input_length = math.ceil(orig_length / scaling_factor)
        x = x[..., :input_length]
        # interpolate to the original length
        x = torch.nn.functional.interpolate(x, size=orig_length, mode='linear')
    elif scaling_factor < 1:
        # downsampling: compute the new length after scaling
        resampled_length = math.ceil(orig_length * scaling_factor)
        # interpolate to the resampled length
        x = torch.nn.functional.interpolate(x, size=resampled_length, mode='linear')
        # pad to match the original length
        padding = orig_length - resampled_length
        x = torch.nn.functional.pad(x, (0, padding), mode='constant', value=0)
    else:
        # scaling_factor == 1, no resampling needed
        pass

    return x

class RandomNoise(RandTransform):
    "Add random white noise to the audio signal."
    order = 10  # (order in which this transform is applied)
    split_idx = 0 # (only use on training data, not validation)
    do_decode = False

    def __init__(self, min_gain=0.0, max_gain=0.001, p=0.5):
        super().__init__(p=p)
        self.min_gain = min_gain
        self.max_gain = max_gain

    def encodes(self, x: TensorAudio):
        device = x.device
        dtype = x.dtype

        gain = random.uniform(self.min_gain, self.max_gain)

        noise = torch.randn_like(x)

        noise = noise * gain

        noisy_waveform = x + noise

        noisy_waveform = noisy_waveform.to(device=device, dtype=dtype)
        noisy_audio = TensorAudio(
            noisy_waveform,
            sample_rate=x.sample_rate,
            cache_path=x.cache_path,
            orig_path=x.orig_path
        )

        return noisy_audio
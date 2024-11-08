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

class RandomResample(RandTransform):
    "Randomly resample the audio to create pitch and speed augmentation, with cached resamplers."
    order = 10  # Order in which this transform is applied
    split_idx = 0
    do_decode = False

    def __init__(
        self,
        min_factor=0.75,              # Minimum scaling factor
        max_factor=1.3,             # Maximum scaling factor
        p=0.5,                       # Probability of applying the transform
        max_resamplers=100,          # Number of resamplers to cache
        lazy=True,                   # Whether to create resamplers lazily
        fixed_orig_freq = 32000,     # Always use this as the basis frequency, regardless of each clip's original sample rate
    ):
        super().__init__(p=p)
        self.min_factor = min_factor
        self.max_factor = max_factor
        
def encodes(self, x: TensorAudio):
    import math

    orig_length = x.shape[-1]
    scaling_factor = random.uniform(self.min_factor, self.max_factor)

    if scaling_factor > 1:
        # Upsampling: compute the required input length and truncate
        input_length = math.ceil(orig_length / scaling_factor)
        x = x[..., :input_length]
        # Interpolate to the original length
        x = torch.nn.functional.interpolate(x, size=orig_length, mode='linear')
    elif scaling_factor < 1:
        # Downsampling: compute the new length after scaling
        resampled_length = math.ceil(orig_length * scaling_factor)
        # Interpolate to the resampled length
        x = torch.nn.functional.interpolate(x, size=resampled_length, mode='linear')
        # Pad to match the original length
        padding = orig_length - resampled_length
        x = torch.nn.functional.pad(x, (0, padding), mode='constant', value=0)
    else:
        # scaling_factor == 1, no resampling needed
        pass

    return x


# class RandomResample(RandTransform):
#     "Randomly resample the audio to create pitch and speed augmentation, with cached resamplers."
#     order = 10  # Order in which this transform is applied
#     split_idx = 0
#     do_decode = False

#     def __init__(
#         self,
#         min_factor=0.85,              # Minimum scaling factor
#         max_factor=1.18,             # Maximum scaling factor
#         p=0.7,                       # Probability of applying the transform
#         max_resamplers=100,          # Number of resamplers to cache
#         lazy=True,                   # Whether to create resamplers lazily
#         fixed_orig_freq = 32000,     # Always use this as the basis frequency, regardless of each clip's original sample rate
#     ):
#         super().__init__(p=p)
#         self.min_factor = min_factor
#         self.max_factor = max_factor
#         self.max_resamplers = max_resamplers
#         self.lazy = lazy
#         self.fixed_orig_freq = fixed_orig_freq

#         # Create an array of scaling factors uniformly distributed between min_factor and max_factor
#         self.scaling_factors = torch.linspace(
#             self.min_factor, self.max_factor, steps=self.max_resamplers
#         ).tolist()

#         self.new_freqs = [int(fixed_orig_freq * scaling_factor) for scaling_factor in self.scaling_factors]

#         self.resamplers = {}

#         # if not self.lazy:
#         #     # Create the resamplers upfront
#         #     for new_freq in self.new_freqs:
#         #         resampler = torchaudio.transforms.Resample(
#         #             orig_freq=fixed_orig_freq, new_freq=new_freq,
#         #             lowpass_filter_width=12, rolloff=0.85, resampling_method="sinc_interp_hann"
#         #         )
#         #         self.resamplers[new_freq] = resampler

#     def encodes(self, x: TensorAudio):
#         # Ensure the tensor is on the correct device
#         device = x.device
#         dtype = x.dtype

#         orig_sample_rate = x.sample_rate

#         # Randomly select one scaling factor
#         new_freq = random.choice(self.new_freqs)

#         # Retrieve or create the resampler
#         resampler = self._get_resampler(self.fixed_orig_freq, new_freq, device=device, dtype=dtype)

#         assert resampler.kernel.device == device, f"Resampler kernel is on {resampler.kernel.device}, expected {device}"

#         print(f"x device: {x.device}, resampler device: {resampler.kernel.device}")
#         print(f"x dtype: {x.dtype}, resampler dtype: {resampler.kernel.dtype}")

#         # Resample the audio
#         # Adjusted to use relative scaling
#         x = resampler(x)

#         # Adjust the length by cropping or padding
#         target_length = x.shape[-1]
#         resampled_length = x.shape[-1]

#         if resampled_length > target_length:
#             # Crop the excess samples
#             x = x[..., :target_length]
#         elif resampled_length < target_length:
#             # Pad with zeros to match the target length
#             padding = target_length - resampled_length
#             x = torch.nn.functional.pad(
#                 x, (0, padding), mode='constant', value=0
#             )

#         return x

#     # def _get_resampler(self, fixed_orig_freq, new_freq, device, dtype):
#     #     "Retrieve or create a resampler for the given scaling factor."
#     #     if new_freq in self.resamplers:
#     #         resampler = self.resamplers[new_freq]
#     #     else:
#     #         # Create the resampler and store it
#     #         resampler = torchaudio.functional.resample(
#     #             orig_freq=fixed_orig_freq, new_freq=new_freq,
#     #             lowpass_filter_width=12, rolloff=0.85, resampling_method="sinc_interp_hann"
#     #         )
#     #         self.resamplers[new_freq] = resampler
#     #         resampler = resampler.to(device=device, dtype=dtype)
#     #         resampler.kernel = resampler.kernel.to(device=device, dtype=dtype)
#     #     return resampler
    

class RandomNoise(RandTransform):
    "Add random white noise to the audio signal."
    order = 10  # Order in which this transform is applied
    split_idx = 0
    do_decode = False

    def __init__(self, min_gain=0.0, max_gain=0.07, p=0.5):
        super().__init__(p=p)
        self.min_gain = min_gain
        self.max_gain = max_gain

    def encodes(self, x: TensorAudio):
        # Ensure the tensor is on the correct device
        device = x.device
        dtype = x.dtype

        # Generate a random gain factor
        gain = random.uniform(self.min_gain, self.max_gain)

        if gain == 0:
            return x  # No noise added

        # Generate white noise
        noise = torch.randn_like(x)

        # Scale the noise by the gain factor
        noise = noise * gain

        # Add noise to the original signal
        noisy_waveform = x + noise

        # Ensure the waveform stays within valid range [-1.0, 1.0]
        noisy_waveform = torch.clamp(noisy_waveform, -1.0, 1.0)

        # Retain the original sample rate and metadata
        noisy_waveform = noisy_waveform.to(device=device, dtype=dtype)
        noisy_audio = TensorAudio(
            noisy_waveform,
            sample_rate=x.sample_rate,
            cache_path=x.cache_path,
            orig_path=x.orig_path
        )

        return noisy_audio
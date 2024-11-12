import sys
from IPython.display import Audio, display
from fastai.vision.data import get_grid
import matplotlib.pyplot as plt
import numpy as np
import torch

# from audiocrush.core import TensorAudioBase

def show_audio(audio_tensor, ctx=None, **kwargs):
    if not hasattr(audio_tensor, 'sample_rate'):
        raise AttributeError("TensorAudio object must have a 'sample_rate' attribute.")
    sr = getattr(audio_tensor, 'sample_rate', 32000)

    # if(audio_tensor.ndim == 3):
    #     audio_tensor = audio_tensor[0]    
    assert audio_tensor.ndim == 2, f"TensorAudio object must have shape: (channels, samples), got {audio_tensor.shape}"

    # TODO: display multichannel in a nice way. For now, mix down to mono:
    audio_tensor = audio_tensor.mean(dim=0)
    
    # convert to 1D tensor:
    audio_tensor = audio_tensor.flatten()
    waveform = audio_tensor.cpu().numpy()

    # Use provided context if available, otherwise create new figure
    if ctx is None: ctx = plt.gca()
    
    ctx.plot(np.arange(len(waveform)) / sr, waveform)
    ctx.set_xlabel('Time (s)')
    ctx.set_ylabel('Amplitude')
    
    # for jupyter, make widget:
    if 'ipykernel' in sys.modules:
        display(Audio(waveform, rate=sr))
    
    return ctx


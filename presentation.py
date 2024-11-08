import sys
from IPython.display import Audio, display
import matplotlib.pyplot as plt
import numpy as np

def show_audio(audio_tensor, ctx=None, **kwargs):
    if not hasattr(audio_tensor, 'sample_rate'):
        raise AttributeError("TensorAudio object must have a 'sample_rate' attribute.")
    sr = getattr(audio_tensor, 'sample_rate', 32000)

    assert audio_tensor.ndim == 2, f"TensorAudio object must have shape: (channels, samples), got {audio_tensor.shape}"

    # TODO: display multichannel in a nice way. For now, mix down to mono:
    audio_tensor = audio_tensor.mean(dim=0)

    # convert to 1D tensor:
    audio_tensor = audio_tensor.flatten()

    waveform = audio_tensor.cpu().numpy()

    # matplotlib plotting - TODO: extend with playback and other visuals:
    plt.figure()
    plt.plot(np.arange(len(waveform)) / sr, waveform)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.show()

    # for jupyter, make widget:
    if 'ipykernel' in sys.modules:
        display(Audio(waveform, rate=sr))

    return ctx
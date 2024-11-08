import subprocess
from fastai.data.transforms import Transform
from fastai.data.block import TransformBlock
from fastai.data.transforms import get_files
from pathlib import Path
from audiocrush.core import TensorAudio

from fastai.torch_basics import *

class AudioTensorCreate(Transform):
    def __init__(self, sample_rate=32000, channels=1, bit_depth=16, integer_format=True, duration = 1.5, cache_dir=None, use_lazy_caching=True):
        self.sample_rate = sample_rate
        self.channels = channels
        self.bit_depth = bit_depth
        self.integer_format = integer_format
        type_prefix = 's' if integer_format else 'f'
        self.encoding_string = f'{type_prefix}{bit_depth}le'
        self.channel_encoding_string = 'planar' if channels > 1 else ''
        self.duration = duration
        self.cache_dir = cache_dir
        self.use_lazy_caching = use_lazy_caching
        if self.cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "audiocrush"
        if not use_lazy_caching:
            pass # TODO

    def encodes(self, orig_path: Path):
        # Check if the file is cached
        cached_file_path = Path(Path(orig_path).name).with_suffix(f'.raw_{self.encoding_string}_{str(self.sample_rate)}hz_{str(self.bit_depth)}bits_{str(self.channels)}ch{self.channel_encoding_string}')
        if not cached_file_path.exists():
            self.cache_file(orig_path, cached_file_path)
        waveform = self.load_complete_file(cached_file_path)
        # print(f"got file: {cached_file_path} { (orig_path)}")
        target_length = int(self.duration * self.sample_rate)
        current_length = waveform.size(1)
        pad_length = max(0, target_length - current_length)
        waveform = torch.nn.functional.pad(waveform, (0, pad_length), mode='constant', value=0)
        waveform = waveform[:,:target_length]
        # Create a TensorAudio object
        return TensorAudio(waveform, sample_rate=self.sample_rate, cached_file_path=cached_file_path, orig_path=orig_path)

    def cache_file(self, orig_path, cached_file_path):
        temp_output_path = str(cached_file_path) + '.tmp'
        command = ['ffmpeg', '-loglevel', 'error', '-i', str(orig_path),
                '-f', self.encoding_string,
                '-acodec', f'pcm_{self.encoding_string}',
                '-ar', str(self.sample_rate),
                '-ac', str(self.channels),
                str(temp_output_path)]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            return None
        os.rename(temp_output_path, cached_file_path)
        # print(f"cached file: {cached_file_path} { (orig_path)}")
        return cached_file_path
    
    def load_complete_file(self, cached_file_path):
        if(self.integer_format and self.bit_depth == 16):
            dtype = np.int16
        elif (not self.integer_format and self.bit_depth == 32):
            dtype = np.float32
        else:
            raise ValueError(f"Unsupported bit depth: {self.bit_depth}")
        memmapped_audio_data = np.memmap(cached_file_path, dtype=dtype, mode='r')
        
        num_samples = len(memmapped_audio_data) // self.channels
        audio_data_np = memmapped_audio_data.reshape(self.channels, num_samples).copy()
        audio_data = torch.from_numpy(audio_data_np).to(torch.float32)
        if (self.integer_format and self.bit_depth == 16):
            audio_data = audio_data / 32768.0
        # print(f"loaded file: {cached_file_path}")
        return audio_data
    
def AudioBlock(sample_rate=32000, cache_dir=None):
    return TransformBlock(type_tfms=AudioTensorCreate(sample_rate=sample_rate, cache_dir=cache_dir))

audio_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('audio/'))

def get_audio_file_names(path, extensions=None, recurse=True, folders=None):
    return get_files(path, extensions=extensions or audio_extensions, recurse=recurse, folders=folders)


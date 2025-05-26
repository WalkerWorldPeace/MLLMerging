import torch
import torchaudio
import torchaudio.compliance.kaldi as ta_kaldi

from moviepy.editor import VideoFileClip
from omegaconf import OmegaConf
import librosa

def preprocess(
    source: torch.Tensor,
    fbank_mean: float = 15.41663,
    fbank_std: float = 6.55582,
) -> torch.Tensor:
    fbanks = []
    for waveform in source:
        waveform = waveform.unsqueeze(0) * 2 ** 15
        print(waveform.shape)
        fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
        fbanks.append(fbank)
    fbank = torch.stack(fbanks, dim=0)
    fbank = (fbank - fbank_mean) / (2 * fbank_std)
    return fbank

def forward_padding_mask(
    features: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    extra = padding_mask.size(1) % features.size(1)
    if extra > 0:
        padding_mask = padding_mask[:, :-extra]
    padding_mask = padding_mask.view(
        padding_mask.size(0), features.size(1), -1
    )
    padding_mask = padding_mask.all(-1)
    return padding_mask

class BeatsAudioProcessor:
    def __init__(self, sampling_rate=16000, n_frames=2, frame_length=512, is_eval=False):
        """
        Adapted from https://github.com/NINAnor/rare_species_detections/blob/main/BEATs/BEATs.py
        """
        super().__init__()

        self.sampling_rate = sampling_rate
        self.n_frames = n_frames
        self.frame_length = frame_length
        self.fbank_mean = 15.41663
        self.fbank_std = 6.55582
        self.is_eval = is_eval

    def _load_audio(self, aupath):
        if aupath.endswith('.mp4'):
            video = VideoFileClip(aupath)
            audio_np = video.audio.to_soundarray(fps=self.sampling_rate)
            if len(audio_np.shape) == 2:
                audio_np = audio_np.mean(axis=1)  # Convert to mono
            waveform = torch.tensor(audio_np).float()
            sr = self.sampling_rate
        else:
            waveform, sr = torchaudio.load(aupath)
            if waveform.shape[0] == 2: 
                waveform = torch.mean(waveform, dim=0)
            if sr != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
                waveform = resampler(waveform)
        return waveform

    def process_audio(self, aupath):
        # audio_inputs, sr = librosa.load(aupath, sr=self.sampling_rate, mono=True)
        # audio_inputs = torch.from_numpy(audio_inputs)
        waveform = self._load_audio(aupath)
        
        # keep maximum 30s
        if len(waveform) > 30 * self.sampling_rate:
            waveform = waveform[:30*self.sampling_rate]
        
        padding_mask = torch.zeros(1, waveform.shape[0]).bool().squeeze(0)
        
        audio_inputs = preprocess(waveform.unsqueeze(0), self.fbank_mean, self.fbank_std)
        padding_mask = forward_padding_mask(audio_inputs, padding_mask.unsqueeze(0))
        audio_inputs, padding_mask = audio_inputs[0], padding_mask[0]
        
        return audio_inputs, padding_mask # N x 128, N

    def __call__(self, aupath, start_sec=None, end_sec=None):
        """
        Args:
            aupath: path to audio file
        Returns:
            torch.tensor: audio clip after transforms.
        """
        # Helper function to return empty tensor for invalid audio
        def empty_audio_tensor():
            return torch.zeros((self.n_frames, self.frame_length, 128))
        
        if isinstance(aupath, list):
            all_audio_inputs = []
            all_audio_padding_mask = []
            for audio_file in aupath:
                audio_frames, padding_mask = self(audio_file)
                all_audio_inputs.append(audio_frames)
                all_audio_padding_mask.append(padding_mask)
            all_audio_inputs = torch.nn.utils.rnn.pad_sequence(all_audio_inputs,
                                                    batch_first=True,
                                                    padding_value=0)
            all_audio_padding_mask = torch.nn.utils.rnn.pad_sequence(all_audio_padding_mask,
                                                            batch_first=True,
                                                            padding_value=1)
            return all_audio_inputs, all_audio_padding_mask
        
        try:
            # Handle MP4 files
            if aupath.endswith('.mp4'):
                video = VideoFileClip(aupath)
                if start_sec is not None and end_sec is not None:
                    video = video.subclip(start_sec, end_sec)
                audio_np = video.audio.to_soundarray(fps=self.sampling_rate)
                if audio_np.ndim == 2:
                    audio_np = audio_np.mean(axis=1)  # Convert to mono
                waveform = torch.tensor(audio_np).float()
                sr = self.sampling_rate
            else:
                waveform, sr = torchaudio.load(aupath)

            # Validate waveform
            if len(waveform.shape) == 0:
                return empty_audio_tensor()

            # Convert stereo to mono
            if waveform.shape[0] == 2:
                waveform = torch.mean(waveform, dim=0)

            # Resample waveform if necessary
            if sr != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
                waveform = resampler(waveform)

        except:
            return empty_audio_tensor()

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        waveform = waveform * 2**15

        # Compute fbank features
        try:
            fbank = ta_kaldi.fbank(
                waveform,
                num_mel_bins=128,
                sample_frequency=self.sampling_rate,
                frame_length=25,
                frame_shift=10,
            )
            fbank = (fbank - self.fbank_mean) / (2 * self.fbank_std)
        except:
            return empty_audio_tensor()

        # Handle padding and frames extraction differently for eval and training modes
        if not self.is_eval:
            fbank_pad_len = self.frame_length * self.n_frames - fbank.shape[0]
            if fbank_pad_len > 0:
                fbank = torch.nn.ZeroPad2d((0, 0, 0, fbank_pad_len))(fbank)
            fbank = fbank[:self.frame_length * self.n_frames]
            frames = [fbank[i*self.frame_length:(i+1)*self.frame_length].unsqueeze(0) for i in range(self.n_frames)]
        else:
            fbank_pad_len = fbank.shape[0] % self.frame_length
            if fbank_pad_len > 0:
                fbank = torch.nn.ZeroPad2d((0, 0, 0, fbank_pad_len))(fbank)
            curr_frames = fbank.shape[0] // self.frame_length
            frames = [fbank[i*self.frame_length:(i+1)*self.frame_length].unsqueeze(0) for i in range(curr_frames)]

        frames = torch.cat(frames, dim=0)
        frames = frames.reshape(-1, frames.shape[-1])
        padding_mask = torch.zeros(1, frames.shape[0]).bool().squeeze(0)
        return frames, padding_mask
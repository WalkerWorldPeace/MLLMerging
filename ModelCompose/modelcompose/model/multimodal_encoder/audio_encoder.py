import torch
import torch.nn as nn

from transformers.modeling_utils import get_parameter_device, get_parameter_dtype
from .beats.BEATs import BEATsConfig, BEATs
from .beats.audio_processor import BeatsAudioProcessor

class BeatsAudioEncoder(nn.Module):
    def __init__(self, audio_encoder, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.audio_encoder_name = audio_encoder

        if not delay_load:
            self.load_model()
        else:
            checkpoint = torch.load(self.audio_encoder_name)
            self.cfg_only = BEATsConfig(checkpoint['cfg'])

    def load_model(self):
        if self.is_loaded:
            return

        checkpoint = torch.load(self.audio_encoder_name)
        self.cfg_only = BEATsConfig(checkpoint['cfg'])
        audio_encoder = BEATs(self.cfg_only)
        audio_encoder.load_state_dict(checkpoint['model'])
        audio_encoder.eval()
        self.audio_encoder = audio_encoder
        
        self.audio_processor = BeatsAudioProcessor()
        
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, audio_inputs, audio_padding_mask=None):
        audio_features, audio_padding_mask = self.audio_encoder.extract_features_new(audio_inputs, audio_padding_mask, feature_only=True)
        return audio_features, ~audio_padding_mask # True for valid tokens

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)

    @property
    def config(self):
        return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.encoder_embed_dim

    @property
    def modal_processor(self):
        return self.audio_processor
    
    @property
    def dummy_inputs(self):
        return {
            'audio_inputs': torch.zeros(1, 1024, 128, device=self.device, dtype=self.dtype),
            'audio_padding_mask': torch.zeros(1, 1024, device=self.device, dtype=self.dtype)
        }
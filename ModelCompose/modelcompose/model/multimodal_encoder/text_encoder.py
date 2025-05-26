import torch
import torch.nn as nn

from transformers import CLIPTextModelWithProjection, CLIPTokenizer, CLIPTextConfig

class CLIPTextProjEncoder(nn.Module):
    def __init__(self, text_encoder, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.text_encoder_name = text_encoder

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPTextConfig.from_pretrained(self.text_encoder_name)

    def load_model(self):
        self.text_tokenizer = CLIPTokenizer.from_pretrained(self.text_encoder_name)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(self.text_encoder_name)
        self.text_encoder.requires_grad_(False)
        
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, input_ids=None, attention_mask=None):
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=self.device)
        text_forward_outs = self.text_encoder(
            input_ids.to(device=self.device),
            attention_mask=attention_mask
        )
        text_features = text_forward_outs.text_embeds

        return text_features.unsqueeze(1)

    @torch.no_grad()
    def encode_inputs(self, texts):
        return self.text_encoder(texts.to(device=self.device, dtype=self.dtype)).text_embeds

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.text_encoder.dtype

    @property
    def device(self):
        return self.text_encoder.device

    @property
    def config(self):
        if self.is_loaded:
            return self.text_encoder.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.projection_dim
    
    @property
    def modal_processor(self):
        return self.text_tokenizer

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
import torch
import torch.nn as nn

from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPVisionConfig

class CLIPVisionProjVisionTower(nn.Module):
    def __init__(self, vision_tower, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModelWithProjection.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)
        
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, pixel_values):
        image_forward_outs = self.vision_tower(pixel_values.to(device=self.device, dtype=self.dtype))
        image_features = image_forward_outs.image_embeds

        return image_features.unsqueeze(1)


    @torch.no_grad()
    def encode_inputs(self, images):
        return self.vision_tower(images.to(device=self.device, dtype=self.dtype)).image_embeds

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.projection_dim
    
    @property
    def modal_processor(self):
        return self.image_processor

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
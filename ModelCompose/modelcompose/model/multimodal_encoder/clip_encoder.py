import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.modeling_utils import get_parameter_device, get_parameter_dtype
from .eva_clip import create_model_and_transforms, create_model_config, MimicCLIPImageProcessor
from .eva_clip_blip import create_eva_vit_g

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            if 'eva' in self.vision_tower_name.lower():
                if 'EVA02_CLIP_L_336_psz14_s6B.pt' in self.vision_tower_name:
                    self.cfg_only = create_model_config('EVA02-CLIP-L-14-336', self.vision_tower_name)
                elif 'EVA01_g_psz14.pt' in self.vision_tower_name:
                    self.cfg_only = create_model_config('EVA01-CLIP-g-14-336', self.vision_tower_name)
            else:
                self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        if 'eva' in self.vision_tower_name.lower():
            if 'EVA02_CLIP_L_336_psz14_s6B.pt' in self.vision_tower_name:
                model, _, preprocess = create_model_and_transforms(
                    'EVA02-CLIP-L-14-336', 
                    self.vision_tower_name, 
                    force_custom_clip=True
                )
                self.image_processor = MimicCLIPImageProcessor(preprocess)
                self.vision_tower = model.visual
                self.vision_tower.requires_grad_(False)
                self.vision_tower.config = create_model_config('EVA02-CLIP-L-14-336', self.vision_tower_name)
            elif 'EVA01_g_psz14.pt' in self.vision_tower_name:
                model, _, preprocess = create_model_and_transforms(
                    'EVA01-CLIP-g-14-336', 
                    self.vision_tower_name, 
                    force_custom_clip=True
                )
                self.image_processor = MimicCLIPImageProcessor(preprocess)
                self.vision_tower = model.visual
                self.vision_tower.requires_grad_(False)
                self.vision_tower.config = create_model_config('EVA01-CLIP-g-14', self.vision_tower_name)
        else:
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
            self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_inputs(self):
        return torch.zeros(1, self.config.num_channels, self.config.image_size, self.config.image_size, device=self.device, dtype=self.dtype)

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
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

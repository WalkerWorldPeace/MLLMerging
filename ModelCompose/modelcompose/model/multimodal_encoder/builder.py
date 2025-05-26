import os
import torch
import torch.nn as nn
from .clip_encoder import CLIPVisionTower

from .vision_encoder import CLIPVisionProjVisionTower
from .text_encoder import CLIPTextProjEncoder
from .imagebind.imagebind_model import imagebind_huge
from .audio_encoder import BeatsAudioEncoder
from .languagebind import LanguageBindImageTower, LanguageBindVideoTower
from .point_encoder import PointEncoder

import einops

# from .video_llama.models.blip2 import Blip2Base
# from .video_llama.models.eva_vit import create_eva_vit_g_with_delay_load, load_eva_vit_g

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

# class EvaClipG(Blip2Base):
#     def __init__(self, model_args, delay_load=False, **kwargs):
#         super().__init__()
#         num_query_token = 32

#         # init VIT
#         self.visual_encoder, self.ln_vision = self.init_vision_encoder(
#             model_name="eva_clip_g", img_size=224, drop_path_rate=0, use_grad_checkpoint=False,precision="fp16", 
#             delay_load=delay_load, ckpt_path=model_args.eva_vit_g_ckpt 
#         )
        
#         # init Qformer
#         self.Qformer, self.query_tokens = self.init_Qformer(
#             num_query_token, self.visual_encoder.num_features
#         )
#         self.Qformer.cls = None
#         self.Qformer.bert.embeddings.word_embeddings = None
#         self.Qformer.bert.embeddings.position_embeddings = None
#         for layer in self.Qformer.bert.encoder.layer:
#             layer.output = None
#             layer.intermediate = None

#         # init audio encoder
#         self.audio_encoder = imagebind_huge(ckpt_path=os.path.join(model_args.mm_audio_encoder, "imagebind_huge.pth"), forward_select=0, **kwargs)
        
#     def forward(self, image):
#         device = image.device
#         image = einops.rearrange(image, 'b c t h w -> (b t) c h w') 
        
#         # embed image features with blip2, out: (b t) q h
#         image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
#         return image_embeds
    
#     @classmethod
#     def init_vision_encoder(
#         cls, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision, delay_load, ckpt_path
#     ):
#         assert model_name == "eva_clip_g", "vit model must be eva_clip_g for current version of MiniGPT-4"
#         visual_encoder = create_eva_vit_g_with_delay_load(
#             img_size, drop_path_rate, use_grad_checkpoint, precision, delay_load, ckpt_path
#         )

#         ln_vision = LayerNorm(visual_encoder.num_features) 
#         return visual_encoder, ln_vision

#     def load_model(self):
#         load_eva_vit_g(self.visual_encoder)
#         self.load_from_pretrained(url_or_filename=q_former_model)
#         self.audio_encoder.load_model()

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if vision_tower.endswith('LanguageBind_Image'):
        return LanguageBindImageTower(vision_tower, args=vision_tower_cfg, cache_dir='./cache_dir', **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

def build_modal_encoders(model_args, **kwargs):
    if getattr(model_args, "mm_vision_encoder", None) is not None:
        model_args.mm_vision_tower = model_args.mm_vision_encoder
        
    modal_encoders = dict()
    if getattr(model_args, "mm_audio_encoder", None) is not None:
        if 'VideoLLaMA' in model_args.mm_audio_encoder:
            modal_encoders["audio"] = imagebind_huge(ckpt_path=os.path.join(model_args.mm_audio_encoder, "imagebind_huge.pth"), **kwargs)
            # Temporary Hacked
            modal_encoders["audio"].modal_processor = nn.Identity()
        else:
            modal_encoders["audio"] = BeatsAudioEncoder(model_args.mm_audio_encoder, model_args, delay_load=kwargs.get('delay_load', False))
    if getattr(model_args, "mm_video_encoder", None) is not None:
        video_encoder = getattr(model_args, 'mm_video_encoder', getattr(model_args, 'video_encoder', None))
        if video_encoder.endswith('LanguageBind_Video_merge'):
            modal_encoders["video"] = LanguageBindVideoTower(video_encoder, args=model_args, cache_dir='./cache_dir', **kwargs)
        else:
            raise ValueError(f'Unknown video encoder: {video_encoder}')
    if getattr(model_args, "mm_point_encoder", None) is not None:
        point_encoder = getattr(model_args, 'mm_point_encoder', getattr(model_args, 'point_encoder', None))
        modal_encoders["point"] = PointEncoder(point_encoder, delay_load=kwargs.get('delay_load', False))
    if getattr(model_args, "mm_vision_tower", None) is not None:
        # modal_encoders["vision"] = CLIPVisionProjVisionTower(model_args.mm_vision_encoder, **kwargs)
        modal_encoders["vision"] = build_vision_tower(model_args, **kwargs)
        # Temporary Hacked
        from transformers import CLIPImageProcessor
        modal_encoders["vision"].modal_processor = CLIPImageProcessor.from_pretrained(model_args.mm_vision_tower)
    if getattr(model_args, "mm_text_encoder", None) is not None:
        modal_encoders["text"] = CLIPTextProjEncoder(model_args.mm_text_encoder, **kwargs)
    
    return nn.ModuleDict(modal_encoders)


def infer_modals(model_args):
    modals = ['default']
    # modals = []
    if getattr(model_args, "mm_audio_encoder", None) is not None:
        modals.append('audio')
    if getattr(model_args, "mm_vision_encoder", None) is not None or getattr(model_args, "mm_vision_tower", None) is not None:
        modals.append('vision')
    if getattr(model_args, "mm_video_encoder", None):
        modals.append('video')
    if getattr(model_args, "mm_point_encoder", None):
        modals.append('point')
    return modals
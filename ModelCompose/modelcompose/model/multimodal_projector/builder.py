import torch
import torch.nn as nn
import re
import sys
import copy

from .Qformer import BertConfig, BertLMHeadModel

import einops

SimpleNamespace = type(sys.implementation)
ModalityType = SimpleNamespace(
    VISION="vision",
    TEXT="text",
    AUDIO="audio",
    THERMAL="thermal",
    DEPTH="depth",
    IMU="imu",
)

class VideoLlamaVideoQformer(nn.Module):
    def __init__(self, model_args, num_query_token=8, vision_width=1024, num_hidden_layers=2):
        super().__init__()

        self.video_Qformer,self.video_query_tokens = self.init_video_Qformer(num_query_token = 32,\
            vision_width=self.Qformer.config.hidden_size, num_hidden_layers =2)
        
        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        
        raise NotImplementedError()
    
        self.audio_Qformer, self.audio_query_tokens = self.init_video_Qformer(num_query_token = num_query_token,\
                vision_width=vision_width, num_hidden_layers =num_hidden_layers)
        
        self.audio_Qformer.cls = None
        self.audio_Qformer.bert.embeddings.word_embeddings = None
        self.audio_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.audio_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None

        self.audio_llama_proj = nn.Linear(
                self.audio_Qformer.config.hidden_size, 4096 
            )
        self.audio_hidden_size = 1024 # see in imagebind_huge()
        self.audio_position_embedding = nn.Embedding(8, self.audio_hidden_size)
        
    def forward(self, x, *args, **kwargs):
        image_embeds = x
        device = x.device
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # add frame_pos embedding
        position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        frame_position_embeddings = self.video_frame_position_embedding(position_ids)
        q_hidden_state = query_output.last_hidden_state

        frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
        frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length)
        frame_hidden_state = frame_position_embeddings + frame_hidden_state

        # frame attention
        frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length)
        frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
        video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)

        video_query_output = self.video_Qformer.bert(
            query_embeds=video_query_tokens,
            encoder_hidden_states=frame_hidden_state,
            encoder_attention_mask=frame_atts,
            return_dict=True,
            )
        video_hidden = video_query_output.last_hidden_state

        inputs_llama = self.llama_proj(video_hidden)
        return inputs_llama
    
    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, num_hidden_layers =2):
            raise NotImplementedError()
            # encoder_config = BertConfig.from_pretrained("bert-base-uncased")
            encoder_config = BertConfig(transformers_version='4.6.0.dev0', gradient_checkpointing=False) # https://huggingface.co/bert-base-uncased/blob/main/config.json
            
            encoder_config.num_hidden_layers = num_hidden_layers
            encoder_config.encoder_width = vision_width
            # insert cross-attention layer every other block
            encoder_config.add_cross_attention = True
            encoder_config.cross_attention_freq = 1
            encoder_config.query_length = num_query_token
            Qformer = BertLMHeadModel(config=encoder_config)
            query_tokens = nn.Parameter(
                torch.zeros(1, num_query_token, encoder_config.hidden_size)
            )
            query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
            return Qformer, query_tokens

class VideoLlamaAudioQformer(nn.Module):
    def __init__(self, num_query_token=8, vision_width=1024, num_hidden_layers=2, num_positions=1024):
        super().__init__()
        self.audio_Qformer, self.audio_query_tokens = self.init_video_Qformer(num_query_token = num_query_token,\
                vision_width=vision_width, num_hidden_layers=num_hidden_layers)
        
        self.audio_Qformer.cls = None
        self.audio_Qformer.bert.embeddings.word_embeddings = None
        self.audio_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.audio_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None

        self.audio_llama_proj = nn.Linear(
                self.audio_Qformer.config.hidden_size, 4096 
            )
        self.audio_hidden_size = vision_width # see in imagebind_huge()
        self.audio_position_embedding = nn.Embedding(num_positions, self.audio_hidden_size)
        
    def forward(self, x, *args, **kwargs):
        audio_imagebind_finalout = x
        device = x.device

        batch_size, time_length = audio_imagebind_finalout.size()[:2]

        position_ids = torch.arange(time_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        audio_position_embeddings = self.audio_position_embedding(position_ids)
        audio_imagebind_finalout = audio_imagebind_finalout + audio_position_embeddings

        audio_query_tokens = self.audio_query_tokens.expand(audio_imagebind_finalout.shape[0], -1, -1)
                
        frame_atts = torch.ones(audio_imagebind_finalout.size()[:-1], dtype=torch.long).to(device)

        audio_query_output = self.audio_Qformer.bert(
            query_embeds=audio_query_tokens, #[32,768]
            encoder_hidden_states=audio_imagebind_finalout,
            encoder_attention_mask=frame_atts,
            return_dict=True,
            )
        audio_hidden = audio_query_output.last_hidden_state

        inputs_llama = self.audio_llama_proj(audio_hidden)
        return inputs_llama
    
    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, num_hidden_layers =2):
            # encoder_config = BertConfig.from_pretrained("bert-base-uncased")
            encoder_config = BertConfig(transformers_version='4.6.0.dev0', gradient_checkpointing=False) # https://huggingface.co/bert-base-uncased/blob/main/config.json
            
            encoder_config.num_hidden_layers = num_hidden_layers
            encoder_config.encoder_width = vision_width
            # insert cross-attention layer every other block
            encoder_config.add_cross_attention = True
            encoder_config.cross_attention_freq = 1
            encoder_config.query_length = num_query_token
            Qformer = BertLMHeadModel(config=encoder_config)
            query_tokens = nn.Parameter(
                torch.zeros(1, num_query_token, encoder_config.hidden_size)
            )
            query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
            return Qformer, query_tokens

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    qformer_match = re.match(r'^qformer_(\d+)N_(\d+)L$', projector_type)
    if qformer_match:
        num_query_token, num_hidden_layers = int(qformer_match.group(1)), int(qformer_match.group(2))
        return VideoLlamaAudioQformer(num_query_token, config.mm_hidden_size, num_hidden_layers)

    if projector_type == 'identity':
        return IdentityMap()
    

    raise ValueError(f'Unknown projector type: {projector_type}')

def build_audio_projector(config, delay_load=False, **kwargs):
    config = copy.deepcopy(config)
    config.mm_projector_type = getattr(config, 'mm_audio_projector_type', 'linear')
    config.mm_hidden_size = getattr(config, 'mm_audio_hidden_size', getattr(config, 'mm_hidden_size', 768))
    return build_vision_projector(config, delay_load, **kwargs)

def build_video_projector(config, delay_load=False, **kwargs):
    config = copy.deepcopy(config)
    config.mm_projector_type = getattr(config, 'mm_video_projector_type', 'linear')
    config.mm_hidden_size = getattr(config, 'mm_video_hidden_size', getattr(config, 'mm_hidden_size', 768))
    return build_vision_projector(config, delay_load, **kwargs)

def build_point_projector(config, delay_load=False, **kwargs):
    config = copy.deepcopy(config)
    config.mm_projector_type = getattr(config, 'mm_point_projector_type', 'linear')
    config.mm_hidden_size = getattr(config, 'mm_point_hidden_size', getattr(config, 'mm_hidden_size', 768))
    return build_vision_projector(config, delay_load, **kwargs)

def build_modal_projectors(config, **kwargs):
    # print(model_args) # DEBUG
    modal_projectors = dict()
    if getattr(config, "mm_audio_encoder", None) is not None:
        if 'VideoLLaMA' in config.mm_audio_encoder:
            modal_projectors['audio'] = VideoLlamaAudioQformer(num_positions=8)
        else:
            modal_projectors['audio'] = build_audio_projector(config)
    if getattr(config, "mm_vision_tower", None) is not None:
        modal_projectors['vision'] = build_vision_projector(config)
    if getattr(config, "mm_video_encoder", None) is not None:
        modal_projectors['video'] = build_video_projector(config)
    if getattr(config, "mm_point_encoder", None) is not None:
        modal_projectors['point'] = build_point_projector(config)
    
    return nn.ModuleDict(modal_projectors)
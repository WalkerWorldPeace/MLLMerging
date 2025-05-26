# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_modal_encoders, infer_modals
from .multimodal_projector.builder import build_vision_projector, build_modal_projectors

from modelcompose.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from modelcompose.constants import (
    IGNORE_INDEX, MODAL_TOKEN_INDEXES
)

import os
import peft

class MultimodalMetaModel:

    def __init__(self, config):
        super(MultimodalMetaModel, self).__init__(config)

        if (
            getattr(config, "mm_hidden_size", None) or
            getattr(config, "mm_audio_hidden_size", None) or 
            getattr(config, "mm_video_hidden_size", None) or 
            getattr(config, "mm_point_hidden_size", None)
        ):
            self.modal_encoders = build_modal_encoders(config, delay_load=True)
            self.modal_projectors = build_modal_projectors(config)

    def get_modal_encoders(self):
        modal_encoders = getattr(self, 'modal_encoders', None)
        if type(modal_encoders) is list:
            modal_encoders = modal_encoders[0]
        return modal_encoders
    
    def get_modal_encoder(self, modal):
        return self.get_modal_encoders()[modal]
    
    def get_modal_projectors(self):
        modal_projectors = getattr(self, 'modal_projectors', None)
        if type(modal_projectors) is list:
            modal_projectors = modal_projectors[0]
        return modal_projectors

    def get_modal_projector(self, modal):
        return self.get_modal_projectors()[modal]

    def initialize_multimodal_modules(self, model_args, fsdp=None):
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        # modal encoder
        self.config.mm_vision_encoder = model_args.mm_vision_tower = model_args.mm_vision_encoder
        self.config.mm_vision_tower = model_args.mm_vision_tower

        if self.get_modal_encoders() is None:
            modal_encoders = build_modal_encoders(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.modal_encoders = [modal_encoders]
            else:
                self.modal_encoders = modal_encoders
        else:
            if fsdp is not None and len(fsdp) > 0:
                modal_encoders = self.modal_encoders[0]
            else:
                modal_encoders = self.modal_encoders
            for modal in modal_encoders:
                modal_encoders[modal].load_model()
        
        if getattr(model_args, 'mm_vision_encoder', None) is not None:
            # we only use this in vision encoder.
            self.config.use_mm_proj = True
            self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
            self.config.mm_hidden_size = self.modal_encoders['vision'].hidden_size
            self.config.mm_vision_select_layer = model_args.mm_vision_select_layer
            self.config.mm_vision_select_feature = model_args.mm_vision_select_feature

        if getattr(model_args, 'mm_audio_encoder', None) is not None:
            self.config.mm_audio_encoder = model_args.mm_audio_encoder
            self.config.mm_audio_projector_type = getattr(model_args, 'mm_audio_projector_type', 'linear')
            self.config.mm_audio_hidden_size = self.modal_encoders['audio'].hidden_size
        
        if getattr(model_args, 'mm_video_encoder', None) is not None:
            self.config.mm_video_encoder = model_args.mm_video_encoder
            self.config.mm_video_projector_type = getattr(model_args, 'mm_video_projector_type', 'linear')
            self.config.mm_video_hidden_size = self.modal_encoders['video'].hidden_size
            self.config.mm_video_select_layer = model_args.mm_video_select_layer
            self.config.mm_video_select_feature = model_args.mm_video_select_feature
        
        if getattr(model_args, 'mm_point_encoder', None) is not None:
            self.config.mm_point_encoder = model_args.mm_point_encoder
            self.config.mm_point_projector_type = getattr(model_args, 'mm_point_projector_type', 'linear')
            self.config.mm_point_hidden_size = self.modal_encoders['point'].hidden_size

        if getattr(self, 'modal_projectors', None) is None:
            # self.modal_projectors = build_modal_projectors(self.config)
            self.modal_projectors = build_modal_projectors(self.config)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            if 'vision' in self.modal_projectors:
                self.modal_projectors['vision'].load_state_dict(get_w(mm_projector_weights, 'modal_projectors.vision'), strict=False)
            
            if 'audio' in self.modal_projectors:
                self.modal_projectors['audio'].load_state_dict(get_w(mm_projector_weights, 'modal_projectors.audio'), strict=False)

            if 'video' in self.modal_projectors:
                self.modal_projectors['video'].load_state_dict(get_w(mm_projector_weights, 'modal_projectors.video'), strict=False)
            
            if 'point' in self.modal_projectors:
                self.modal_projectors['point'].load_state_dict(get_w(mm_projector_weights, 'modal_projectors.point'), strict=False)
        
        if getattr(self.modal_projectors, 'audio', None) is not None:
            audio_projector = self.modal_projectors['audio']
            if 'VideoLLaMA' in model_args.mm_audio_encoder:
                ckpt_path = os.path.join(model_args.mm_audio_encoder, "AL_LLaMA_2_7B_Pretrained.pth")
                if os.path.isfile(ckpt_path):
                    # Load from original ckpt
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                    audio_projector.load_state_dict(ckpt['model'], strict=False)

                    for name, param in audio_projector.named_parameters():
                        print(f"Parameter: {name}, Require Gradient: {param.requires_grad}")
        
        def convert_dict(old_dict, maxsplit=-1):
            new_dict = {}
            for key, value in old_dict.items():
                keys = key.split('.', maxsplit=maxsplit)
                current_dict = new_dict
                for k in keys[:-1]:
                    current_dict.setdefault(k, {})
                    current_dict = current_dict[k]
                current_dict[keys[-1]] = value
            return new_dict
        
        if getattr(model_args, 'projectors_path', None) is not None:
            # Load projectors from saved ckpt
            ckpt_path = os.path.join(model_args.projectors_path)
            if os.path.isfile(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location="cpu")
                if 'base_model.model.model.modal_projectors.audio.audio_query_tokens' in ckpt.keys():
                    # lora-training, load from non_lora_trainables.bin
                    ckpt = convert_dict(ckpt, maxsplit=3)['base_model']['model']['model']
                else:
                    ckpt = convert_dict(ckpt, maxsplit=1)['model']
                self.load_state_dict(ckpt, strict=False)


class MultimodalMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_modal_encoders()

    def get_modal_encoders(self):
        return self.get_model().get_modal_encoders()

    def get_modal_encoder(self, modal):
        return self.get_model().get_modal_encoder(modal)
    
    def get_modal_projectors(self):
        return self.get_model().get_modal_projectors()

    def get_modal_projector(self, modal):
        return self.get_model().get_modal_projector(modal)
    
    def get_modal_processors(self):
        modal_encoders = self.get_modal_encoders()
        modal_processors = dict()
        for k in modal_encoders:
            modal_processors[k] = modal_encoders[k].modal_processor
        return modal_processors

    def encode_modal_inputs(self, inputs, prefix_tokens=None, suffix_tokens=None):
        all_modal_features = dict()
        all_modal_attention_mask = dict()
        modals = [modal for modal in infer_modals(self.config) if modal != 'default']
        for modal in modals:
            encoder, projector = self.get_model().get_modal_encoder(modal), self.get_model().get_modal_projector(modal)
            if modal not in inputs:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                modal_inputs = encoder.dummy_inputs
            else:
                modal_inputs = inputs[modal]
            attention_mask = None
            if type(modal_inputs) is list:
                if modal == 'audio':
                    # [ImageBind]
                    # pre-stack
                    # concat_inputs = torch.cat([inp for inp in modal_inputs], dim=0)
                    stack_inputs = torch.stack(modal_inputs, dim=0)
                    input_features = encoder(stack_inputs)
                    input_features = projector(input_features)
                    # split_sizes = [inp.shape[0] for inp in inputs]
                    # features = torch.split(input_features, split_sizes, dim=0)
                    # features = [x.flatten(0, 1) for x in features]

                    features = input_features
                else:
                    # pre-concat
                    concat_inputs = torch.cat([inp for inp in modal_inputs], dim=0)
                    input_features = encoder(concat_inputs)
                    input_features = projector(input_features)
                    split_sizes = [inp.shape[0] for inp in inputs]
                    features = torch.split(input_features, split_sizes, dim=0)
                    features = [x.flatten(0, 1) for x in features]
            else:
                if modal == 'audio':
                    # features, attention_mask = encoder(**modal_inputs)
                    features, _ = encoder(**modal_inputs)
                    features = projector(features)
                elif modal == 'video':
                    features = encoder(modal_inputs) # b x t x n x d
                    b, t, n, d = features.shape
                    features = features.reshape(b, t*n, d)
                    features = projector(features)
                else:
                    features = encoder(modal_inputs)
                    features = projector(features)
            
            # add prefix and suffix tokens
            b = features.shape[0]
            all_features = []
            if prefix_tokens is not None and modal in prefix_tokens:
                all_features.append(prefix_tokens[modal].expand(b, -1, -1))
            all_features.append(features)
            if suffix_tokens is not None and modal in suffix_tokens:
                all_features.append(suffix_tokens[modal].expand(b, -1, -1))
            features = torch.cat(all_features, dim=1)
            
            # update attention mask
            if attention_mask is not None:
                all_attention_mask = []
                if prefix_tokens is not None and modal in prefix_tokens:
                    all_attention_mask.append(torch.ones(b, prefix_tokens[modal].shape[1]).to(attention_mask))
                all_attention_mask.append(attention_mask)
                if suffix_tokens is not None and modal in suffix_tokens:
                    all_attention_mask.append(torch.ones(b, suffix_tokens[modal].shape[1]).to(attention_mask))
                attention_mask = torch.cat(all_attention_mask, dim=1)
            else:
                attention_mask = torch.ones(b, features.shape[1]).to(features.device)
            all_modal_features[modal] = features
            all_modal_attention_mask[modal] = attention_mask
        return all_modal_features, all_modal_attention_mask

    def modal_token_match(self, input_ids):
        matched = dict()
        
        for modal, token in MODAL_TOKEN_INDEXES.items():
            token_matched = input_ids == token
            if token_matched.sum() != 0:
                matched[modal] = token_matched
        
        modal, start_index = None, 10000
        for k, v in matched.items():
            start_index_k = torch.where(v)[0][0]
            if start_index_k < start_index:
                start_index = start_index_k
                modal = k
                
        return modal, start_index

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, modal_inputs, prefix_tokens, suffix_tokens
    ):
        if modal_inputs is None or input_ids.shape[1] == 1:
            if past_key_values is not None and modal_inputs is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels, None
        
        # import torch.distributed as dist
        # if 'video' in modal_inputs:
        #     print(dist.get_rank(), modal_inputs['video'].shape)
        modal_features, modal_features_attention_mask = self.encode_modal_inputs(modal_inputs, prefix_tokens, suffix_tokens)
        
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_modal_idx = {modal: 0 for modal in MODAL_TOKEN_INDEXES}
        modal_attention_mask = dict()
        # for modal_t in modal_inputs:
        #     modal_attention_mask[modal_t] = []
        for modal_t in modal_features:
            modal_attention_mask[modal_t] = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_new_input_embeds = []
            cur_modal_attention_mask = dict()
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            
            # attention mask for each modality
            for modal_t in modal_inputs:
                cur_modal_attention_mask[modal_t] = []
            
            # find next modal token in cur_input_ids
            modal, modal_token_start = self.modal_token_match(cur_input_ids) # str, int
            
            if modal is None:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                
                cur_modal_features = []
                for modal in modal_features:
                    cur_modal_features.append(modal_features[modal][0])
                cur_modal_features = torch.cat(cur_modal_features, dim=0)
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_modal_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                
                for modal in modal_features:
                    modal_attention_mask[modal].append(torch.full((len(cur_input_embeds),), False, dtype=attention_mask.dtype, device=attention_mask.device))
                
                continue
            
            while modal is not None:
                cur_modal_features = modal_features[modal][cur_modal_idx[modal]]
                cur_modal_features_attention_mask = modal_features_attention_mask[modal][cur_modal_idx[modal]] # N_modal
                cur_modal_features_attention_mask = cur_modal_features_attention_mask.to(dtype=attention_mask.dtype, device=attention_mask.device) # N_modal
                
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:modal_token_start]))
                cur_new_input_embeds.append(cur_modal_features)
                
                # Add Modal Attention Mask
                for modal_t in cur_modal_attention_mask:
                    if modal_t != modal:
                        cur_modal_attention_mask[modal_t].append(torch.full((modal_token_start+len(cur_modal_features),), False, dtype=attention_mask.dtype, device=attention_mask.device))
                    else:
                        cur_modal_attention_mask[modal_t].append(torch.full((modal_token_start,), False, dtype=attention_mask.dtype, device=attention_mask.device))
                        # cur_modal_attention_mask[modal_t].append(torch.full((len(cur_modal_features),), True, dtype=attention_mask.dtype, device=attention_mask.device))
                        cur_modal_attention_mask[modal_t].append(cur_modal_features_attention_mask)
                
                if labels is not None:
                    cur_new_labels.append(cur_labels[:modal_token_start])
                    cur_new_labels.append(torch.full((cur_modal_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[modal_token_start+1:]
                cur_modal_idx[modal] += 1
                cur_input_ids = cur_input_ids[modal_token_start+1:]
                modal, modal_token_start = self.modal_token_match(cur_input_ids) # str, int

            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                # Add Modal Attention Mask
                for modal_t in cur_modal_attention_mask:
                    cur_modal_attention_mask[modal_t].append(torch.full((len(cur_input_ids),), False, dtype=attention_mask.dtype, device=attention_mask.device))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
                    
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            
            # Add Modal Attention Mask
            for modal_t in cur_modal_attention_mask:
                cur_modal_attention_mask[modal_t] = torch.cat(cur_modal_attention_mask[modal_t], dim=0)
                modal_attention_mask[modal_t].append(cur_modal_attention_mask[modal_t])
            
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)
            
            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            # Add Modal Attention Mask
            new_modal_attention_mask = dict()
            for model_t in modal_attention_mask:
                new_modal_attention_mask[model_t] = []
                for cur_modal_attention_mask in modal_attention_mask[model_t]:
                    cur_modal_attention_mask = torch.cat((cur_modal_attention_mask, torch.full((max_len - cur_modal_attention_mask.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)), dim=0)
                    new_modal_attention_mask[model_t].append(cur_modal_attention_mask)
                new_modal_attention_mask[model_t] = torch.stack(new_modal_attention_mask[model_t], dim=0) 
            modal_attention_mask = new_modal_attention_mask
                    
            # for cur_modal_attention_mask in modal_attention_mask:
            #     cur_modal_attention_mask = torch.cat((cur_modal_attention_mask, torch.full((max_len - cur_modal_attention_mask.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)), dim=0)
            #     new_modal_attention_mask.append(cur_modal_attention_mask)
            # modal_attention_mask = torch.stack(new_modal_attention_mask, dim=0)
        
            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            
            # Add Modal Attention Mask
            # modal_attention_mask = torch.stack(modal_attention_mask, dim=0)
            new_modal_attention_mask = dict()
            for model_t in modal_attention_mask:
                if len(modal_attention_mask[model_t]):
                    new_modal_attention_mask[model_t] = torch.stack(modal_attention_mask[model_t], dim=0) 
            modal_attention_mask = new_modal_attention_mask
            
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), 
                                           dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        # add original mask for text tokens
        if len(modal_attention_mask):
            modal_attention_mask['default'] = (torch.sum(torch.stack([modal_attention_mask[k] for k in modal_attention_mask]), dim=0) == 0)
            # if not (modal_attention_mask['default']==False).any():
            #     modal_attention_mask = None
        else:
            modal_attention_mask = None
            
        return None, attention_mask, past_key_values, new_input_embeds, new_labels, modal_attention_mask
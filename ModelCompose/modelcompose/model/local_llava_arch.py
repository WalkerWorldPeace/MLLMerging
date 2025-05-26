from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch

from .llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from modelcompose.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

class LocalLlavaMetaForCausalLM(LlavaMetaForCausalLM):
    
    def get_video_encoder(self):
        return self.get_model().get_video_encoder()
    def get_audio_encoder(self):
        return self.get_model().get_audio_encoder()

    def encode_videos(self, videos):
        video_features = self.get_video_encoder()(videos)
        return video_features
    
    def encode_audios(self, audios):
        audio_features = self.get_audio_encoder()(audios)
        return audio_features
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, modal_attention_mask,
        prefix_tokens, suffix_tokens
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
                
            return input_ids, attention_mask, past_key_values, None, labels, None, None

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)
            
            # [LOCAL TUNABLE TOKENS]
            b = image_features.shape[0]
            all_features = []
            local_attn_mask = []
            if prefix_tokens is not None:
                all_features.append(prefix_tokens.expand(b, -1, -1))
                local_attn_mask.append(torch.ones(b, prefix_tokens.shape[1]).to(prefix_tokens))
            all_features.append(image_features)
            local_attn_mask.append(torch.zeros(b, image_features.shape[1]).to(image_features))
            if suffix_tokens is not None:
                all_features.append(suffix_tokens.expand(b, -1, -1))
                local_attn_mask.append(torch.ones(b, suffix_tokens.shape[1]).to(suffix_tokens))
            image_features = torch.cat(all_features, dim=1)
            local_attn_mask = torch.cat(local_attn_mask, dim=1)
        
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        modal_attention_mask = []
        localtoken_attention_mask = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            cur_modal_attention_mask = []
            cur_localtoken_attention_mask = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                cur_local_attn_mask = local_attn_mask[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                    
                    # Add Modal Attention Mask
                    cur_modal_attention_mask.append(torch.full((image_token_start-1,), False, dtype=attention_mask.dtype, device=attention_mask.device))
                    cur_modal_attention_mask.append(torch.full((2+len(cur_image_features),), True, dtype=attention_mask.dtype, device=attention_mask.device))
                    
                    
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                        cur_labels = cur_labels[image_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    
                    # Add Modal Attention Mask
                    cur_modal_attention_mask.append(torch.full((image_token_start,), False, dtype=attention_mask.dtype, device=attention_mask.device))
                    cur_modal_attention_mask.append(torch.full((len(cur_image_features),), True, dtype=attention_mask.dtype, device=attention_mask.device))
                    
                    # Add Localtoken Attention Mask
                    cur_localtoken_attention_mask.append(torch.full((image_token_start,), False, dtype=attention_mask.dtype, device=attention_mask.device))
                    cur_localtoken_attention_mask.append(cur_local_attn_mask)
                    
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                # Add Modal Attention Mask
                cur_modal_attention_mask.append(torch.full((len(cur_input_ids),), False, dtype=attention_mask.dtype, device=attention_mask.device))
                # Add Localtoken Attention Mask
                cur_localtoken_attention_mask.append(torch.full((len(cur_input_ids),), False, dtype=attention_mask.dtype, device=attention_mask.device))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            # Add Modal Attention Mask
            cur_modal_attention_mask = torch.cat(cur_modal_attention_mask, dim=0)
            # Add Localtoken Attention Mask
            cur_localtoken_attention_mask = torch.cat(cur_localtoken_attention_mask, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            
            # Add Modal Attention Mask
            modal_attention_mask.append(cur_modal_attention_mask)
            # Add Localtoken Attention Mask
            localtoken_attention_mask.append(cur_localtoken_attention_mask)
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
            new_modal_attention_mask = []
            for cur_modal_attention_mask in modal_attention_mask:
                cur_modal_attention_mask = torch.cat((cur_modal_attention_mask, torch.full((max_len - cur_modal_attention_mask.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)), dim=0)
                new_modal_attention_mask.append(cur_modal_attention_mask)
            modal_attention_mask = torch.stack(new_modal_attention_mask, dim=0)

            # Add Localtoken Attention Mask
            new_localtoken_attention_mask = []
            for cur_localtoken_attention_mask in localtoken_attention_mask:
                cur_localtoken_attention_mask = torch.cat((cur_localtoken_attention_mask, torch.full((max_len - cur_localtoken_attention_mask.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)), dim=0)
                new_localtoken_attention_mask.append(cur_localtoken_attention_mask)
            localtoken_attention_mask = torch.stack(new_localtoken_attention_mask, dim=0)
            
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
            modal_attention_mask = torch.stack(modal_attention_mask, dim=0)
            
            # Add Localtoken Attention Mask
            localtoken_attention_mask = torch.stack(localtoken_attention_mask, dim=0)

            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels, modal_attention_mask, localtoken_attention_mask
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


from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..multimodal_arch import MultimodalMetaModel, MultimodalMetaForCausalLM, infer_modals

from modelcompose.constants import IGNORE_INDEX

class MultimodalConfig(LlamaConfig):
    model_type = "multimodal"
    lora_strategy = None
    lora_name = "default"
    lora_r = 128
    lora_alpha = 256
    lora_dropout = 0.05
    local_prefix_tokens = 0
    local_suffix_tokens = 0
    local_vision_prefix_tokens = None
    local_vision_suffix_tokens = None
    local_audio_prefix_tokens = None
    local_audio_suffix_tokens = None
    local_video_prefix_tokens = None
    local_video_suffix_tokens = None
    local_point_prefix_tokens = None
    local_point_suffix_tokens = None
    
    layer_local_tokens = False
    seperate_layernorm = False
    
    merge_default_weights = None
    reset_scaling_weights = None
    
    # [Multimodal]
    mm_vision_encoder = None
    mm_audio_encoder = None
    mm_video_encoder = None
    mm_point_encoder = None


from peft.tuners.lora import Linear as LoraLinear
from peft.utils import transpose
from transformers.models.llama.modeling_llama import *

class LocalLoraLinear(LoraLinear):
    
    def __init__(
        self, 
        adapter_names: List[str],
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        merge_default_weights: str = None,
        reset_scaling_weights: str = None,
        **kwargs
    ):
        adapter_name = adapter_names[0]
        init_lora_weights = kwargs.get("init_lora_weights", True)
        super().__init__(adapter_name, in_features, out_features, r, lora_alpha, lora_dropout, fan_in_fan_out, is_target_conv_1d_layer, **kwargs)
        for adapter_name in adapter_names[1:]:
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        
        self.merge_default_weights = merge_default_weights
        # [MERGE] on-the-fly adjust scale
        self.reset_scaling_weights = reset_scaling_weights
        # print(self.merge_default_weights)
        # print(self.reset_scaling_weights)
        # print(adapter_names)
        if self.reset_scaling_weights is not None:
            reset_scaling = self.extract_params(self.reset_scaling_weights)
            if any(['default-' in key for key in reset_scaling]):
                self.merge_default_weights = 'linear-'
                self.default_adapter_names = [f"default-{adapter_name}" for adapter_name in adapter_names[1:]] 
                # self.default_adapter_weights = {n:1 for n in self.default_adapter_names} # DO NOT set this, set self.scaling instead
                for default_adapter_name in self.default_adapter_names:
                    self.update_layer(default_adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            
            for key in reset_scaling:
                if key in self.scaling:
                    self.scaling[key] = self.scaling[key] * reset_scaling[key]
            
            self.reset_scaling = reset_scaling
            self.adaptive_weights = None

    def extract_params(self, input_string):
        # Split the parameter section by ',' to get key-value pairs
        param_pairs = input_string.split(',')
        # Initialize a dictionary to store the parameters
        params = {}
        # Iterate through the key-value pairs
        for pair in param_pairs:
            key, value = pair.split('=')
            params[key.strip()] = float(value)
        return params

    def forward(self, x: torch.Tensor, active_adapters=None):
        previous_dtype = x.dtype
        original_outputs = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        # outputs = {'original': original_outputs}
        if active_adapters:
            outputs = dict()
            for active_adapter in active_adapters:
                # print(active_adapter)
                if active_adapter not in self.lora_A:
                    outputs[active_adapter] = original_outputs
                    continue
                if active_adapter == 'default' and self.merge_default_weights is not None:
                    sub_outputs = []
                    for default_adapter_name in self.default_adapter_names:
                        x = x.to(self.lora_A[default_adapter_name].weight.dtype)
                        sub_outputs.append(
                            self.lora_B[default_adapter_name](
                                self.lora_A[default_adapter_name](self.lora_dropout[default_adapter_name](x))
                            )
                            * self.scaling[default_adapter_name]
                        )
                    if self.merge_default_weights == 'sum':
                        lora_branch = torch.stack(sub_outputs).sum(0) # naive sum
                    elif self.merge_default_weights == 'mean':
                        lora_branch = torch.stack(sub_outputs).mean(0) # naive mean
                    elif self.merge_default_weights.startswith('linear-'): # linear interpolation
                        lora_branch = torch.stack(sub_outputs).sum(0)
                    else:
                        raise NotImplementedError(f"online merging strategy '{self.merge_default_weights}' is not implemented.")
                    outputs['default'] = (original_outputs + lora_branch).to(previous_dtype)
                    continue
                x = x.to(self.lora_A[active_adapter].weight.dtype)
                lora_branch = (
                    self.lora_B[active_adapter](
                        self.lora_A[active_adapter](self.lora_dropout[active_adapter](x))
                    )
                    * self.scaling[active_adapter]
                )
                # print(active_adapter,2)
                # print(lora_branch)
                outputs[active_adapter] = (original_outputs + lora_branch).to(previous_dtype)
            return outputs
        else:
            return original_outputs

class LocalLoraAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = config.pretraining_tp
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.modal_names = infer_modals(config)
        # [MODAL ATTENTION]
        self.q_proj = LocalLoraLinear(self.modal_names, self.hidden_size, self.num_heads * self.head_dim, config.lora_r, config.lora_alpha, config.lora_dropout, bias=False, reset_scaling_weights=self.config.reset_scaling_weights)
        self.k_proj = LocalLoraLinear(self.modal_names, self.hidden_size, self.num_key_value_heads * self.head_dim, config.lora_r, config.lora_alpha, config.lora_dropout, bias=False, reset_scaling_weights=self.config.reset_scaling_weights)
        self.v_proj = LocalLoraLinear(self.modal_names, self.hidden_size, self.num_key_value_heads * self.head_dim, config.lora_r, config.lora_alpha, config.lora_dropout, bias=False, reset_scaling_weights=self.config.reset_scaling_weights)
        self.o_proj = LocalLoraLinear(self.modal_names, self.num_heads * self.head_dim, self.hidden_size, config.lora_r, config.lora_alpha, config.lora_dropout, bias=False, reset_scaling_weights=self.config.reset_scaling_weights)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        modal_attention_mask: Optional[Dict[str, torch.Tensor]] = None, # [MODAL ATTENTION], {modal: (bsz, q_len)}
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
            
        else:
            name = '+'.join(self.modal_names)
            if modal_attention_mask is None:
                query_states = self.q_proj(hidden_states, active_adapters=('default',))['default']
                key_states = self.k_proj(hidden_states, active_adapters=('default',))['default']
                value_states = self.v_proj(hidden_states, active_adapters=('default',))['default']
                
                # name = 'lora-'+name
                # states_tensor = torch.repeat_interleave(torch.cat([query_states, key_states, value_states]).norm(dim=-1), 5, dim=0).unsqueeze(0)
                # save_with_incrementing_filename(states_tensor.clone().detach(), f'/yeesuanAI05/thumt/cc/MITv2/LLaVA_1021/playground/analysis/{name}', 'states')
            else:
                # [MODAL ATTENTION]

                # lora_adapter_names = modal_attention_mask.keys()
                # query_states, query_states_lora = self.q_proj(hidden_states, active_adapters=lora_adapter_names)
                # key_states, key_states_lora = self.k_proj(hidden_states, active_adapters=lora_adapter_names)
                # value_states, value_states_lora = self.v_proj(hidden_states, active_adapters=lora_adapter_names)

                # query_states = query_states * modal_attention_mask['original']
                # key_states = key_states * modal_attention_mask['original']
                # value_states = value_states = modal_attention_mask['original']

                # for idx, adapter_name in enumerate(lora_adapter_names):
                #     pass
                # print(2,self.modal_names)
                query_states_mapping = self.q_proj(hidden_states, active_adapters=self.modal_names)
                key_states_mapping = self.k_proj(hidden_states, active_adapters=self.modal_names)
                value_states_mapping = self.v_proj(hidden_states, active_adapters=self.modal_names)

                query_states = torch.stack([query_states_mapping[k] * modal_attention_mask[k].unsqueeze(-1).to(hidden_states) for k in modal_attention_mask]).sum(dim=0)
                key_states = torch.stack([key_states_mapping[k] * modal_attention_mask[k].unsqueeze(-1).to(hidden_states) for k in modal_attention_mask]).sum(dim=0)
                value_states = torch.stack([value_states_mapping[k] * modal_attention_mask[k].unsqueeze(-1).to(hidden_states) for k in modal_attention_mask]).sum(dim=0)
                
                # name = 'pdt-'+name
                # states_tensor = torch.repeat_interleave(torch.cat([query_states, key_states, value_states]).norm(dim=-1), 5, dim=0).unsqueeze(0)
                # save_with_incrementing_filename(states_tensor.clone().detach(), f'/yeesuanAI05/thumt/cc/MITv2/LLaVA_1021/playground/analysis/{name}', 'states')

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            if modal_attention_mask is None:
                attn_output = self.o_proj(attn_output, active_adapters=('default',))['default']
            else:
                # [MODAL ATTENTION]
                # attn_output, attn_output_lora = self.o_proj(attn_output, active_adapters=(self.config.lora_name,))
                # attn_output = attn_output * (1.0 - modal_attention_mask) + attn_output_lora * modal_attention_mask
                
                attn_output_mapping = self.o_proj(attn_output, active_adapters=self.modal_names)
                attn_output = torch.stack([attn_output_mapping[k] * modal_attention_mask[k].unsqueeze(-1).to(hidden_states) for k in modal_attention_mask]).sum(dim=0)
        
        # index_number = save_with_incrementing_filename(attn_weights.clone().detach().mean(dim=1)**0.2, f'/yeesuanAI05/thumt/cc/MITv2/LLaVA_1021/playground/analysis/{name}', 'attn_weights')
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LocalLoraMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pretraining_tp = config.pretraining_tp
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        self.modal_names = infer_modals(config)
        # [MODAL ATTENTION]
        self.gate_proj = LocalLoraLinear(self.modal_names, self.hidden_size, self.intermediate_size, config.lora_r, config.lora_alpha, config.lora_dropout, bias=False, reset_scaling_weights=config.reset_scaling_weights)
        self.up_proj = LocalLoraLinear(self.modal_names, self.hidden_size, self.intermediate_size, config.lora_r, config.lora_alpha, config.lora_dropout, bias=False, reset_scaling_weights=config.reset_scaling_weights)
        self.down_proj = LocalLoraLinear(self.modal_names, self.intermediate_size, self.hidden_size, config.lora_r, config.lora_alpha, config.lora_dropout, bias=False, reset_scaling_weights=config.reset_scaling_weights)
        
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, modal_attention_mask=None): # [MODAL ATTENTION]
        # if modal_attention_mask is not None:
        #     modal_attention_mask = modal_attention_mask.unsqueeze(-1).to(x) # [MODAL ATTENTION], (b, n, 1), 1 for modal inputs
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat([F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)]
            down_proj = sum(down_proj)
        else:
            # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            if modal_attention_mask is not None:
                gated_outputs = self.gate_proj(x, active_adapters=self.modal_names)
                for modal_t in gated_outputs:
                    gated_outputs[modal_t] = self.act_fn(gated_outputs[modal_t])
                
                up_proj_outputs = self.up_proj(x, active_adapters=self.modal_names)
                down_proj_outputs = dict()
                for modal_t in up_proj_outputs:
                    down_proj_outputs[modal_t] = self.down_proj(gated_outputs[modal_t] * up_proj_outputs[modal_t], active_adapters=[modal_t])[modal_t]
                
                down_proj = torch.stack([down_proj_outputs[k] * modal_attention_mask[k].unsqueeze(-1).to(x) for k in modal_attention_mask]).sum(dim=0)
            else:
                gated_outputs = self.act_fn(self.gate_proj(x, active_adapters=('default',))['default'])
                up_proj_outputs = self.up_proj(x, active_adapters=('default',))['default']
                down_proj = self.down_proj(gated_outputs * up_proj_outputs, active_adapters=('default',))['default']

        return down_proj

class MultimodalLlamaDecoderLayer(nn.Module):
    
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LocalLoraAttention(config=config)
        self.mlp = LocalLoraMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        modal_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # [MODAL ATTENTION]
        if past_key_value is not None:
            # No modal tokens during generation
            modal_attention_mask = None
        
        # Self Attention        
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            modal_attention_mask=modal_attention_mask, # [MODAL ATTENTION]
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states, modal_attention_mask) # [MODAL ATTENTION]
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MultimodalLlamaModel(MultimodalMetaModel, LlamaModel):
    config_class = MultimodalConfig

    def __init__(self, config: MultimodalConfig):
        # super(LlavaLlamaModel, self).__init__(config)
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([MultimodalLlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        modal_attention_mask: Optional[torch.Tensor] = None, # [MODAL ATTENTION]
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    modal_attention_mask, # [MODAL ATTENTION]
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    modal_attention_mask=modal_attention_mask, # [MODAL ATTENTION]
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # exit()

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MultimodalLlamaForCausalLM(LlamaForCausalLM, MultimodalMetaForCausalLM):
    config_class = MultimodalConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MultimodalLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.modal_names = infer_modals(config)
        # [MODAL ATTENTION]
        self.prefix_tokens, self.suffix_tokens = None, None
        if config.local_prefix_tokens != 0:
            self.prefix_tokens = {}
            for modal_t in self.modal_names:
                modal_prefix_tokens = getattr(config, f"local_{modal_t}_prefix_tokens", None)
                if modal_prefix_tokens is None:
                    modal_prefix_tokens = config.local_prefix_tokens
                self.prefix_tokens[modal_t] = nn.Parameter(torch.zeros(1, modal_prefix_tokens, config.hidden_size))
            self.prefix_tokens = nn.ParameterDict(self.prefix_tokens)
        if config.local_suffix_tokens != 0:
            self.suffix_tokens = {}
            for modal_t in self.modal_names:
                modal_suffix_tokens = getattr(config, f"local_{modal_t}_suffix_tokens", None)
                if modal_suffix_tokens is None:
                    modal_suffix_tokens = config.local_suffix_tokens
                self.suffix_tokens[modal_t] = nn.Parameter(torch.zeros(1, modal_suffix_tokens, config.hidden_size))
            self.suffix_tokens = nn.ParameterDict(self.suffix_tokens)

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        if "modal_encoders" in module.__class__.__name__:
            return
        elif isinstance(module, LocalLoraLinear):
            from transformers.deepspeed import is_deepspeed_zero3_enabled
            if is_deepspeed_zero3_enabled():
                import deepspeed
                weights_to_init = [t for k, t in module.named_parameters() if "lora_" in k]
                with deepspeed.zero.GatheredParameters(weights_to_init, modifier_rank=0):
                    adapter_keys = module.lora_A.keys()
                    for adapter_key in adapter_keys:
                        module.reset_lora_parameters(adapter_key)
            else:
                adapter_keys = module.lora_A.keys()
                for adapter_key in adapter_keys:
                    module.reset_lora_parameters(adapter_key)
        else:
            super()._init_weights(module)

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        modal_inputs: Optional[Dict[str, torch.FloatTensor]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # import ipdb; ipdb.set_trace()
        if input_ids is not None:
            input_ids, attention_mask, past_key_values, inputs_embeds, labels, modal_attention_mask = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, modal_inputs, self.prefix_tokens, self.suffix_tokens)
        # otherwise, inputs_embeds is provided.

        # import ipdb; ipdb.set_trace()

        # Use modal_attention_mask=None to disable LocalLoRA
        if self.config.lora_strategy not in ['modal', 'modal+language']:
            modal_attention_mask = None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            modal_attention_mask=modal_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "modal_inputs": kwargs.get("modal_inputs", None),
            }
        )
        return model_inputs

AutoConfig.register("multimodal", MultimodalConfig)
AutoModelForCausalLM.register(MultimodalConfig, MultimodalLlamaForCausalLM)

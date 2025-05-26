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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.models.llama.modeling_llama import *
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.utils import logging

from peft.tuners.lora import Linear as LoraLinear
from peft.utils import transpose
from ..local_llava_arch import LlavaMetaModel, LocalLlavaMetaForCausalLM

logger = logging.get_logger(__name__)

class LocalLlavaConfig(LlamaConfig):
    model_type = "localllava"
    local_lora_enable = False
    lora_name = "default"
    lora_r = 128
    lora_alpha = 256
    lora_dropout = 0.05
    local_prefix_tokens = 0
    local_suffix_tokens = 0
    layer_local_tokens = False
    seperate_layernorm = False

from typing import Optional, Tuple
import warnings

import torch

import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
except ImportError:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as flash_attn_unpadded_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input

class LocalLoraLinear(LoraLinear):
    
    def forward(self, x: torch.Tensor, active_adapters=None):
        previous_dtype = x.dtype
        original_outputs = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        outputs = [original_outputs]
        
        if active_adapters:
            for active_adapter in active_adapters:
                x = x.to(self.lora_A[active_adapter].weight.dtype)
                lora_branch = (
                    self.lora_B[active_adapter](
                        self.lora_A[active_adapter](self.lora_dropout[active_adapter](x))
                    )
                    * self.scaling[active_adapter]
                )
                outputs.append((original_outputs + lora_branch).to(previous_dtype))
            return outputs
        else:
            return original_outputs

class LocalLlavaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LocalLlavaConfig):
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
        
        # self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        # self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # [MODAL ATTENTION]
        self.q_proj = LocalLoraLinear(config.lora_name, self.hidden_size, self.num_heads * self.head_dim, config.lora_r, config.lora_alpha, config.lora_dropout, bias=False)
        self.k_proj = LocalLoraLinear(config.lora_name, self.hidden_size, self.num_key_value_heads * self.head_dim, config.lora_r, config.lora_alpha, config.lora_dropout, bias=False)
        self.v_proj = LocalLoraLinear(config.lora_name, self.hidden_size, self.num_key_value_heads * self.head_dim, config.lora_r, config.lora_alpha, config.lora_dropout, bias=False)
        self.o_proj = LocalLoraLinear(config.lora_name, self.num_heads * self.head_dim, self.hidden_size, config.lora_r, config.lora_alpha, config.lora_dropout, bias=False)
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

    # # from llama_flash_attn_monkey_patch.py
    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     modal_attention_mask: Optional[torch.Tensor] = None, # [MODAL ATTENTION]
    #     position_ids: Optional[torch.Tensor] = None,
    #     past_key_value: Optional[Tuple[torch.Tensor]] = None,
    #     output_attentions: bool = False,
    #     use_cache: bool = False,
    # ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    #     if output_attentions:
    #         warnings.warn(
    #             "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
    #         )

    #     bsz, q_len, _ = hidden_states.size()
    #     modal_attention_mask = modal_attention_mask.unsqueeze(-1).to(hidden_states) # [MODAL ATTENTION], (b, n, 1), 1 for modal inputs
        
    #     # query_states = (
    #     #     self.q_proj(hidden_states)
    #     #     .view(bsz, q_len, self.num_heads, self.head_dim)
    #     #     .transpose(1, 2)
    #     # )
    #     # key_states = (
    #     #     self.k_proj(hidden_states)
    #     #     .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    #     #     .transpose(1, 2)
    #     # )
    #     # value_states = (
    #     #     self.v_proj(hidden_states)
    #     #     .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    #     #     .transpose(1, 2)
    #     # )  # shape: (b, num_heads, s, head_dim)
        
    #     # [MODAL ATTENTION]
    #     self.q_proj.disable_adapters = True
    #     self.k_proj.disable_adapters = True
    #     self.v_proj.disable_adapters = True
        
    #     query_states = self.q_proj(hidden_states)
    #     key_states = self.k_proj(hidden_states)
    #     value_states = self.v_proj(hidden_states)
        
    #     self.q_proj.disable_adapters = False
    #     self.k_proj.disable_adapters = False
    #     self.v_proj.disable_adapters = False
        
    #     query_states_lora = self.q_proj(hidden_states)
    #     key_states_lora = self.k_proj(hidden_states)
    #     value_states_lora = self.v_proj(hidden_states)
        
    #     # import torch.distributed as dist
    #     # if dist.get_rank() == 0:
    #     #     import ipdb; ipdb.set_trace()
    #     # dist.barrier()
        
    #     query_states = query_states * (1.0 - modal_attention_mask) + query_states_lora * modal_attention_mask
    #     key_states = key_states * (1.0 - modal_attention_mask) + key_states_lora * modal_attention_mask
    #     value_states = value_states * (1.0 - modal_attention_mask) + value_states_lora * modal_attention_mask

    #     query_states = (
    #         query_states
    #         .view(bsz, q_len, self.num_heads, self.head_dim)
    #         .transpose(1, 2)
    #     )
    #     key_states = (
    #         key_states
    #         .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    #         .transpose(1, 2)
    #     )
    #     value_states = (
    #         value_states
    #         .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    #         .transpose(1, 2)
    #     )  # shape: (b, num_heads, s, head_dim)
        

    #     kv_seq_len = key_states.shape[-2]
    #     if past_key_value is not None:
    #         kv_seq_len += past_key_value[0].shape[-2]

    #     cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    #     query_states, key_states = apply_rotary_pos_emb(
    #         query_states, key_states, cos, sin, position_ids
    #     )

    #     if past_key_value is not None:
    #         # reuse k, v
    #         key_states = torch.cat([past_key_value[0], key_states], dim=2)
    #         value_states = torch.cat([past_key_value[1], value_states], dim=2)

    #     past_key_value = (key_states, value_states) if use_cache else None

    #     # repeat k/v heads if n_kv_heads < n_heads
    #     key_states = repeat_kv(key_states, self.num_key_value_groups)
    #     value_states = repeat_kv(value_states, self.num_key_value_groups)

    #     # Transform the data into the format required by flash attention
    #     qkv = torch.stack([query_states, key_states, value_states], dim=2)
    #     qkv = qkv.transpose(1, 3)  # shape: [b, s, 3, num_heads, head_dim]
    #     key_padding_mask = attention_mask
        
    #     if key_padding_mask is None:
    #         qkv = qkv.reshape(-1, 3, self.num_heads, self.head_dim)
    #         cu_q_lens = torch.arange(
    #             0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device
    #         )
    #         max_s = q_len
    #         output = flash_attn_unpadded_qkvpacked_func(
    #             qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
    #         )
    #         output = output.view(bsz, q_len, -1)
    #     else:
    #         qkv = qkv.reshape(bsz, q_len, -1)
    #         qkv, indices, cu_q_lens, max_s = unpad_input(qkv, key_padding_mask)
    #         qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
    #         output_unpad = flash_attn_unpadded_qkvpacked_func(
    #             qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
    #         )
    #         output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
    #         output = pad_input(output_unpad, indices, bsz, q_len)

    #     # attn_output = self.o_proj(attn_output)
            
    #     # [MODAL ATTENTION]
    #     self.o_proj.disable_adapters = True
    #     attn_output = self.o_proj(output)
        
    #     self.o_proj.disable_adapters = False
    #     attn_output_lora = self.o_proj(output)
        
    #     attn_output = attn_output * (1.0 - modal_attention_mask) + attn_output_lora * modal_attention_mask

    #     return attn_output, None, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        modal_attention_mask: Optional[torch.Tensor] = None, # [MODAL ATTENTION], (bsz, q_len)
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
            if modal_attention_mask is None:
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)
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
                

                query_states, query_states_lora = self.q_proj(hidden_states, active_adapters=(self.config.lora_name,))
                key_states, key_states_lora = self.k_proj(hidden_states, active_adapters=(self.config.lora_name,))
                value_states, value_states_lora = self.v_proj(hidden_states, active_adapters=(self.config.lora_name,))
                
                modal_attention_mask = modal_attention_mask.unsqueeze(-1).to(hidden_states) # (b, q_len, 1), 1 for modal inputs
                
                query_states = query_states * (1.0 - modal_attention_mask) + query_states_lora * modal_attention_mask
                key_states = key_states * (1.0 - modal_attention_mask) + key_states_lora * modal_attention_mask
                value_states = value_states * (1.0 - modal_attention_mask) + value_states_lora * modal_attention_mask   

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
                attn_output = self.o_proj(attn_output)
            else:
                # [MODAL ATTENTION]
                attn_output, attn_output_lora = self.o_proj(attn_output, active_adapters=(self.config.lora_name,))
                attn_output = attn_output * (1.0 - modal_attention_mask) + attn_output_lora * modal_attention_mask
            
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    

class LocalLlavaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pretraining_tp = config.pretraining_tp
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        # [MODAL ATTENTION]
        self.gate_proj = LoraLinear(config.lora_name, self.hidden_size, self.intermediate_size, config.lora_r, config.lora_alpha, config.lora_dropout, bias=False)
        self.up_proj = LoraLinear(config.lora_name, self.hidden_size, self.intermediate_size, config.lora_r, config.lora_alpha, config.lora_dropout, bias=False)
        self.down_proj = LoraLinear(config.lora_name, self.intermediate_size, self.hidden_size, config.lora_r, config.lora_alpha, config.lora_dropout, bias=False)
        
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, modal_attention_mask=None): # [MODAL ATTENTION]
        if modal_attention_mask is not None:
            modal_attention_mask = modal_attention_mask.unsqueeze(-1).to(x) # [MODAL ATTENTION], (b, n, 1), 1 for modal inputs
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
                # [MODAL ATTENTION]
                self.gate_proj.disable_adapters = True
                self.up_proj.disable_adapters = True
                self.down_proj.disable_adapters = True
                
                down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
                
                self.gate_proj.disable_adapters = False
                self.up_proj.disable_adapters = False
                self.down_proj.disable_adapters = False
                
                down_proj_lora = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
                
                down_proj = down_proj * (1.0 - modal_attention_mask) + down_proj_lora * modal_attention_mask
            else:
                self.gate_proj.disable_adapters = True
                self.up_proj.disable_adapters = True
                self.down_proj.disable_adapters = True
                
                down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LocalLlavaDecoderLayer(nn.Module):
    
    def __init__(self, config: LocalLlavaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LocalLlavaAttention(config=config)
        self.mlp = LocalLlavaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.seperate_layernorm:
            self.image_input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.image_post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # [MODAL ATTENTION]
        self.local_tokens = None
        if config.layer_local_tokens:
            self.local_tokens = nn.Parameter(torch.zeros(1, config.local_prefix_tokens + config.local_suffix_tokens, config.hidden_size))
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        modal_attention_mask: Optional[torch.Tensor] = None,
        localtoken_attention_mask: Optional[torch.Tensor] = None,
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

        if self.local_tokens is not None and localtoken_attention_mask is not None:
            n_modal_inputs = int(localtoken_attention_mask.sum()) // self.local_tokens.shape[1]
            hidden_states[localtoken_attention_mask.bool()] = self.local_tokens.repeat(n_modal_inputs, 1, 1).reshape(-1, self.local_tokens.shape[-1])

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # [MODAL ATTENTION]
        if past_key_value is not None:
            # No modal tokens during generation
            modal_attention_mask = None
        
        # [MODAL ATTENTION]
        modal_attention_mask_unsqueeze = None
        if modal_attention_mask is not None:
            modal_attention_mask_unsqueeze = modal_attention_mask.unsqueeze(-1).to(hidden_states)
        
        if hasattr(self, "image_input_layernorm") and modal_attention_mask_unsqueeze is not None:
            image_hidden_states = self.image_input_layernorm(residual)
            hidden_states = hidden_states * (1.0 - modal_attention_mask_unsqueeze) + image_hidden_states * modal_attention_mask_unsqueeze
        
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
        
        # [MODAL ATTENTION]
        if hasattr(self, "image_post_attention_layernorm") and modal_attention_mask_unsqueeze is not None:
            # modal_attention_mask = modal_attention_mask.unsqueeze(-1).to(hidden_states)
            image_hidden_states = self.image_post_attention_layernorm(residual)
            hidden_states = hidden_states * (1.0 - modal_attention_mask_unsqueeze) + image_hidden_states * modal_attention_mask_unsqueeze
        
        hidden_states = self.mlp(hidden_states, modal_attention_mask) # [MODAL ATTENTION]
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LocalLlavaModel(LlavaMetaModel, LlamaModel):
    config_class = LocalLlavaConfig

    def __init__(self, config: LocalLlavaConfig):
        # super(LlavaLlamaModel, self).__init__(config)
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LocalLlavaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

        self.video_encoder = None
        self.audio_encoder = None
    
    def get_video_encoder(self):
        video_encoder = getattr(self, 'video_encoder', None)
        if type(video_encoder) is list:
            video_encoder = video_encoder[0]
        return video_encoder
    def get_audio_encoder(self):
        audio_encoder = getattr(self, 'audio_encoder', None)
        if type(audio_encoder) is list:
            audio_encoder = audio_encoder[0]
        return audio_encoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        modal_attention_mask: Optional[torch.Tensor] = None, # [MODAL ATTENTION]
        localtoken_attention_mask: Optional[torch.Tensor] = None, # [MODAL ATTENTION]
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
                    localtoken_attention_mask, # [MODAL ATTENTION]
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    modal_attention_mask=modal_attention_mask, # [MODAL ATTENTION]
                    localtoken_attention_mask=localtoken_attention_mask, # [MODAL ATTENTION]
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

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LocalLlavaForCausalLM(LlamaForCausalLM, LocalLlavaMetaForCausalLM):
    config_class = LocalLlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LocalLlavaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # [MODAL ATTENTION]
        self.prefix_tokens, self.suffix_tokens = None, None
        if config.local_prefix_tokens != 0:
            self.prefix_tokens = nn.Parameter(torch.zeros(1, config.local_prefix_tokens, config.hidden_size))
        if config.local_suffix_tokens != 0:
            self.suffix_tokens = nn.Parameter(torch.zeros(1, config.local_suffix_tokens, config.hidden_size))
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        modal_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels, modal_attention_mask, localtoken_attention_mask = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images, modal_attention_mask, self.prefix_tokens, self.suffix_tokens)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            modal_attention_mask=modal_attention_mask, # [MODAL ATTENTION]
            localtoken_attention_mask=localtoken_attention_mask, # [MODAL ATTENTION]
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

        # import torch.distributed as dist
        # if dist.get_rank() == 0:
        #     import ipdb; ipdb.set_trace()
        # dist.barrier()

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
                "modal_attention_mask": kwargs.get("modal_attention_mask", None), # [MODAL ATTENTION]
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("localllava", LocalLlavaConfig)
AutoModelForCausalLM.register(LocalLlavaConfig, LocalLlavaForCausalLM)

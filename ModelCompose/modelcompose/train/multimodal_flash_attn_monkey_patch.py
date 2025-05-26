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

import modelcompose

# from llama_flash_attn_monkey_patch.py
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    modal_attention_mask: Optional[torch.Tensor] = None, # [MODAL ATTENTION]
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()
    
    if modal_attention_mask is None:
        query_states = self.q_proj(hidden_states, active_adapters=('default',))['default']
        key_states = self.k_proj(hidden_states, active_adapters=('default',))['default']
        value_states = self.v_proj(hidden_states, active_adapters=('default',))['default']
    else:
        query_states_mapping = self.q_proj(hidden_states, active_adapters=self.modal_names)
        key_states_mapping = self.k_proj(hidden_states, active_adapters=self.modal_names)
        value_states_mapping = self.v_proj(hidden_states, active_adapters=self.modal_names)

        query_states = torch.stack([query_states_mapping[k] * modal_attention_mask[k].unsqueeze(-1).to(hidden_states) for k in query_states_mapping]).sum(dim=0)
        key_states = torch.stack([key_states_mapping[k] * modal_attention_mask[k].unsqueeze(-1).to(hidden_states) for k in key_states_mapping]).sum(dim=0)
        value_states = torch.stack([value_states_mapping[k] * modal_attention_mask[k].unsqueeze(-1).to(hidden_states) for k in value_states_mapping]).sum(dim=0)
        
    query_states = (
        query_states
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        key_states
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        value_states
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )  # shape: (b, num_heads, s, head_dim)
    

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        # reuse k, v
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Transform the data into the format required by flash attention
    qkv = torch.stack([query_states, key_states, value_states], dim=2)
    qkv = qkv.transpose(1, 3)  # shape: [b, s, 3, num_heads, head_dim]
    key_padding_mask = attention_mask
    
    if key_padding_mask is None:
        qkv = qkv.reshape(-1, 3, self.num_heads, self.head_dim)
        cu_q_lens = torch.arange(
            0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device
        )
        max_s = q_len
        output = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
        output = output.view(bsz, q_len, -1)
    else:
        qkv = qkv.reshape(bsz, q_len, -1)
        qkv, indices, cu_q_lens, max_s = unpad_input(qkv, key_padding_mask)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
        output_unpad = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
        output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
        output = pad_input(output_unpad, indices, bsz, q_len)
        
    # [MODAL ATTENTION]
    if modal_attention_mask is None:
        attn_output = self.o_proj(output, active_adapters=('default',))['default']
    else:
        # [MODAL ATTENTION]
        # attn_output, attn_output_lora = self.o_proj(attn_output, active_adapters=(self.config.lora_name,))
        # attn_output = attn_output * (1.0 - modal_attention_mask) + attn_output_lora * modal_attention_mask
        
        attn_output_mapping = self.o_proj(output, active_adapters=self.modal_names)
        attn_output = torch.stack([attn_output_mapping[k] * modal_attention_mask[k].unsqueeze(-1).to(hidden_states) for k in attn_output_mapping]).sum(dim=0)

    return attn_output, None, past_key_value


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask


def replace_llava_attn_with_flash_attn():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        warnings.warn(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )
    modelcompose.model.language_model.multimodal_llama.MultimodalLlamaModel._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask
    )
    modelcompose.model.language_model.multimodal_llama.LocalLoraAttention.forward = forward

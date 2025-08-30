import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable


from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    ACT2FN,
    LlamaAttention,
    Cache, 
    Unpack, 
    FlashAttentionKwargs,
    apply_rotary_pos_emb,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
    logger
)

from train.train_parameter import FakeQuantizer
from train.config import QuantizeConfig

class LlamaMLPWithR4(nn.Module):
    def __init__(self, module: LlamaMLP, R4):
        super().__init__()
        self.config = module.config
        self.hidden_size = module.hidden_size
        self.intermediate_size = module.intermediate_size
        self.gate_proj = module.gate_proj
        self.up_proj = module.up_proj
        self.down_proj = module.down_proj
        self.act_fn = module.act_fn
        self.R4 = R4

    def forward(self, x):
        # We modify (add R4)
        gated_activation = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        gated_activation_dtype = gated_activation.dtype
        gated_activation_device = gated_activation.device
        down_proj = self.down_proj((gated_activation.to(dtype = self.R4.weight.dtype) @ self.R4.weight.to(gated_activation_device)).to(dtype = gated_activation_dtype))
        return down_proj
    

class LlamaAttentionWithR3(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, module: LlamaAttention, R3, k_quant_config: QuantizeConfig, v_quant_config: QuantizeConfig):
        super().__init__()
        self.config = module.config
        self.layer_idx = module.layer_idx
        self.head_dim = module.head_dim
        self.num_key_value_groups = module.num_key_value_groups
        self.scaling = module.scaling
        self.attention_dropout = module.attention_dropout
        self.is_causal = module.is_causal
        self.q_proj = module.q_proj
        self.k_proj = module.k_proj
        self.v_proj = module.v_proj
        self.o_proj = module.o_proj

        # Optional Rotation Matrix
        self.R3 = R3
        self.k_quant_config = k_quant_config
        self.v_quant_config = v_quant_config
        self.kQuant = None
        self.vQuant = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # We modify (add R3)
        q_type = query_states.dtype
        k_type = key_states.dtype
        q_device = query_states.device
        k_device = key_states.device
        query_states = query_states.to(dtype = self.R3.weight.dtype) @ self.R3.weight.to(device=q_device)
        key_states = key_states.to(dtype = self.R3.weight.dtype) @ self.R3.weight.to(device=k_device)
        query_states = query_states.to(dtype=q_type)
        key_states = key_states.to(dtype=k_type)
        

        # Key:
        if self.kQuant is None:   # First time setting the quantizer
            self.kQuant = FakeQuantizer(self.k_quant_config, key_states)
        else:
            key_states = self.kQuant(key_states)

        # Value:
        if self.vQuant is None:   # First time setting the quantizer
            self.vQuant = FakeQuantizer(self.v_quant_config, value_states)
        else:
            value_states = self.vQuant(value_states)


        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    

def get_parent_module(model, module_name):
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

def apply_R3R4_change_llama_model(model, R3_list, R4_list, k_Quant_config: QuantizeConfig, v_Quant_config: QuantizeConfig):
    """
        Replace LlamaMLP with LlamaMLPWithR4
        Replace LlamaAttention with LlamaAttentionWithR3
    """
    attn_layer_idx = 0  # Record which layer of LlamaAttention it is.
    mlp_layer_idx = 0   # Record which layer of LlamaMLP it is.
    for name, module in model.named_modules():
        if isinstance(module, LlamaMLP):
            parent, attr_name = get_parent_module(model, name)

            # Take out the R4 of the corresponding layer from the list.
            R4_layer = R4_list[mlp_layer_idx]
            setattr(parent, attr_name, LlamaMLPWithR4(module, R4_layer))
            mlp_layer_idx += 1
        elif isinstance(module, LlamaAttention):
            parent, attr_name = get_parent_module(model, name)

            # Take out the R3 of the corresponding layer from the list.
            R3_layer = R3_list[attn_layer_idx]
            setattr(parent, attr_name, LlamaAttentionWithR3(module, R3_layer, k_Quant_config, v_Quant_config))
            attn_layer_idx += 1
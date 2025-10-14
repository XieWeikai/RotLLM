import torch
import torch.nn as nn
import copy
from typing import Tuple, Optional
import torch.nn.functional as F
import math

from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaAttention,
    logger,
    Cache,
    apply_rotary_pos_emb,
    repeat_kv
)


from train.config import QuantizeConfig
from train.train_parameter import FakeQuantizer
from utils.hadamard_utils import get_hadK, matmul_hadU_cuda, hadamard_transform

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
        
        
        # assert gated_activation.shape[-1] == self.intermediate_size, f"Expected last dim {self.intermediate_size}, but got {gated_activation.shape[-1]}"
        # had_K, K = get_hadK(self.intermediate_size)

        # gated_activation = matmul_hadU_cuda(gated_activation, had_K, K).to(dtype = gated_activation_dtype)
        # down_proj = self.down_proj(gated_activation)

        return down_proj


class LlamaAttentionWithR3(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, module: LlamaAttention, R3, k_quant_config: QuantizeConfig, v_quant_config: QuantizeConfig, to_quant: bool = True):
        super().__init__()
        self.config = module.config
        self.layer_idx = module.layer_idx
        if self.layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = module.attention_dropout
        self.hidden_size = module.hidden_size
        self.num_heads = module.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = module.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = module.max_position_embeddings
        self.rope_theta = module.rope_theta
        self.is_causal = module.is_causal

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.q_proj = module.q_proj
        self.k_proj = module.k_proj
        self.v_proj = module.v_proj
        self.o_proj = module.o_proj

        self.k_quant_config = copy.deepcopy(k_quant_config)
        self.v_quant_config = copy.deepcopy(v_quant_config)

        # Optional Rotation Matrix
        self.R3 = R3
        if to_quant:
            self.kQuant = FakeQuantizer(self.k_quant_config)
            self.vQuant = FakeQuantizer(self.v_quant_config)
        else:
            self.kQuant = None
            self.vQuant = None

        # TODO (joao): remove in v4.45 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = module.rotary_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # We modify (add R3)
        q_type = query_states.dtype
        k_type = key_states.dtype
        q_device = query_states.device
        k_device = key_states.device
        query_states = query_states.to(dtype = self.R3.weight.dtype) @ self.R3.weight.to(device=q_device)
        key_states = key_states.to(dtype = self.R3.weight.dtype) @ self.R3.weight.to(device=k_device)

        # query_states = hadamard_transform(query_states.float()) / math.sqrt(query_states.shape[-1])
        # key_states = hadamard_transform(key_states.float()) / math.sqrt(key_states.shape[-1])


        query_states = query_states.to(dtype=q_type)
        key_states = key_states.to(dtype=k_type)
        
        # Transpose: To unify the second dimension of the input parameter scale of StaticLearnableFakeQuantizeFunction as seqlen
        # In order to uniformly perform truncation on this dimension in StaticLearnableFakeQuantizeFunction
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        # Key:
        if self.kQuant is not None:
            key_states = self.kQuant(key_states)

        # Value:
        if self.vQuant is not None:
            value_states = self.vQuant(value_states)

        # Transpose again: to prevent affecting subsequent calculations
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    

def get_parent_module(model, module_name):
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

def apply_R3R4_change_model(model, R3_list, R4_list, k_Quant_config: QuantizeConfig, v_Quant_config: QuantizeConfig, to_quant: bool = True):
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
            setattr(parent, attr_name, LlamaAttentionWithR3(module, R3_layer, k_Quant_config, v_Quant_config, to_quant))
            attn_layer_idx += 1


def value_kv_quantizers(model: nn.Module):
    """
    Traverse all LlamaAttentionWithR3 layers in the model, 
    and use the saved k_Quant_config and v_Quant_config 
    to assign values to kQuant and vQuant.
    """
    for module in model.modules():
        if isinstance(module, LlamaAttentionWithR3):
            assert module.kQuant is None and module.vQuant is None, "Reassign kQuant or vQuant"
            module.kQuant = FakeQuantizer(module.k_quant_config)
            module.vQuant = FakeQuantizer(module.v_quant_config)
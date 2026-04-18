from typing import Iterable
import torch
from torch import nn
from .utils import get_text_tower


def fuse_ln_linear(
    layernorm: torch.nn.Module, linear_layers: Iterable[torch.nn.Linear]
) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias"):
            if linear.bias is None: 
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W_, layernorm.bias.double()
            )
            linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_layer_norms(model):
   
    _, text_model = get_text_tower(model)
    lm_head = model.get_output_embeddings()

    # 从动态获取的文本模型主体中提取 layers 和 final norm
    layers = text_model.layers
    final_norm = text_model.norm

    # 3. 执行融合操作
    for layer in layers:
        # fuse the input layernorms into the linear layers
        fuse_ln_linear(
            layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj]
        )
        fuse_ln_linear(
            layer.input_layernorm,
            [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ],
        )

        W_norm = layer.post_attention_layernorm.weight.data
        layer.post_attention_layernorm.weight.data = torch.ones_like(W_norm)
        W_norm = layer.input_layernorm.weight.data
        layer.input_layernorm.weight.data = torch.ones_like(W_norm)

    # 融合 final norm 到 lm_head
    fuse_ln_linear(
        final_norm,
        [lm_head],
    )
    W_norm = final_norm.weight.data
    final_norm.weight.data = torch.ones_like(W_norm)
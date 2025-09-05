import torch
from torch import nn
from typing import Union
from transformers import Phi3ForCausalLM  

from ..common import RotateOperationRegistry, AutoOperation, NormLinearIterator
from ..rotation_utils import fuse_layer_norms


@NormLinearIterator.register_iterator
class Phi3NormLinearIterator(NormLinearIterator):
    def __init__(self, model: Phi3ForCausalLM):
        super().__init__()
        self.model = model

    def __iter__(self):
        for layer in self.model.model.layers:
            yield layer, "input_layernorm", [layer.self_attn.qkv_proj]
            yield layer, "post_attention_layernorm", [layer.mlp.gate_up_proj]
        yield self.model.model, "norm", [self.model.lm_head]

    @classmethod
    def supports_model(cls, model: nn.Module) -> bool:
        return isinstance(model, Phi3ForCausalLM)


@torch.inference_mode()
def rotate_model(model: Phi3ForCausalLM, R: torch.Tensor, R_v: list[torch.Tensor] = None):
    config = model.config
    dim = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = dim // num_heads
    num_layers = config.num_hidden_layers

    assert R.shape == (dim, dim)

    if isinstance(R_v, torch.Tensor):
        assert R_v.shape == (head_dim, head_dim)
        R_v = [R_v for _ in range(num_layers)]

    assert R_v is None or len(R_v) == num_layers
    assert all([R_v[i].shape == (head_dim, head_dim) for i in range(num_layers)]) if R_v else True

    AutoOperation.rotate_output(model.model.embed_tokens, R)

    for l, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        AutoOperation.rotate_input(attn.qkv_proj, R.T)

        AutoOperation.rotate_output(attn.o_proj, R)

        if R_v is not None:
            AutoOperation.rotate_attn_v(attn, R_v[l])

        mlp = layer.mlp
        AutoOperation.rotate_input(mlp.gate_up_proj, R.T)
        AutoOperation.rotate_output(mlp.down_proj, R)

    AutoOperation.rotate_input(model.lm_head, R.T)


@RotateOperationRegistry.register(Phi3ForCausalLM)
def apply_untie_word_embeddings(model: Phi3ForCausalLM, *args, **kwargs):
    if model.config.tie_word_embeddings:
        print("Untie word embeddings")
        model.config.tie_word_embeddings = False 
        new_weight = model.model.embed_tokens.weight.clone()
        model.lm_head.weight = nn.Parameter(new_weight)
        assert model.model.embed_tokens.weight.data_ptr() != model.lm_head.weight.data_ptr()

@RotateOperationRegistry.register(Phi3ForCausalLM)
def apply_fuse_layer_norms(model: Phi3ForCausalLM, *args, **kwargs):
    print("Fuse layer norms")
    fuse_layer_norms(model)

@RotateOperationRegistry.register(Phi3ForCausalLM)
def apply_rotate_model(model: Phi3ForCausalLM, *args, **kwargs):
    print("Rotate Phi3 model")
    rotate_model(model, *args, **kwargs)

try:
    from transformers.models.phi3.modeling_phi3 import Phi3Attention

    @AutoOperation.register_operation("rotate_attn_v", Phi3Attention)
    def op_rotate_attn_v_phi3(attn: Phi3Attention, R_v: torch.Tensor):
        config = attn.config
        num_heads = config.num_attention_heads
        dim = config.hidden_size
        head_dim = dim // num_heads

        qkv_weight = attn.qkv_proj.weight  # [3 * dim, dim]
        q_proj, k_proj, v_proj = qkv_weight.chunk(3, dim=0)

        v_proj = v_proj.view(num_heads, head_dim, dim).to(dtype=torch.float64)
        v_proj = torch.matmul(R_v.T.unsqueeze(0), v_proj).to(dtype=qkv_weight.dtype)
        v_proj = v_proj.view(-1, dim)

        qkv_weight_rot = torch.cat([q_proj, k_proj, v_proj], dim=0)
        attn.qkv_proj.weight.data.copy_(qkv_weight_rot)

        AutoOperation.rotate_input(attn.o_proj, torch.block_diag(*([R_v] * num_heads)).T)

except ImportError:
    print("Warning: Phi3Attention not found.")

print("----Phi3 rotate module imported----")

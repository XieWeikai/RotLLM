import torch
from torch import nn
from typing import Union
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention
from ..common import RotateOperationRegistry
from ..common import AutoOperation
from ..common import NormLinearIterator


@NormLinearIterator.register_iterator
class LlamaNormLinearIterator(NormLinearIterator):
    def __init__(self, model: LlamaForCausalLM):
        super().__init__()
        self.model = model
        
    def __iter__(self):
        for layer in self.model.model.layers:
            yield layer, "input_layernorm", [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ]
            yield layer, "post_attention_layernorm", [
                layer.mlp.up_proj,
                layer.mlp.gate_proj,
            ]
        yield self.model.model, "norm", [self.model.lm_head]
        
    @classmethod
    def supports_model(cls, model: nn.Module) -> bool:
        return isinstance(model, LlamaForCausalLM)


@AutoOperation.register_operation("rotate_attn_v", LlamaAttention)
def op_rotate_attn_v_for_LM(
    attn: LlamaAttention,
    R_v: torch.Tensor):
    """
    rotate the v (one of the inputs of attention) by a rotation matrix R_v 
    and rotate v back before W_o
    """
    config = attn.config
    num_qo_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    
    # rotate v in attention
    # i.e. rotate the output of W_v
    # note that the output is something like [v_1, v_2, ..., v_{num_heads}]
    # where v_i is a head_dim vector
    # so we need to rotate each head
    # results should be something like [v_1R_v, v_2R_v, ..., v_{num_heads}R_v]
    # this is equal to [v_1, v_2, ..., v_{num_heads}] @ diag(R_v, R_v, ..., R_v) (num_heads times)
    # so we need to rotate the output of W_v by diag(R_v, R_v, ..., R_v)
    R_v_rot = torch.block_diag(*([R_v] * num_kv_heads))
    # rotate_linear_output([attn.v_proj], R_v_rot)
    AutoOperation.rotate_output(attn.v_proj, R_v_rot)
    
    # then we need to rotate back the input of W_o
    # since o_i is linear combination of v_i
    # we can rotate the o_i by R_v^T to get back the original o_i
    # rotate_linear_input([attn.o_proj], torch.block_diag(*([R_v] * num_qo_heads)).T)
    AutoOperation.rotate_input(attn.o_proj, torch.block_diag(*([R_v] * num_qo_heads)).T)


def untie_word_embeddings(model):
    if model.config.tie_word_embeddings:
        # Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
        # this is because the weight of RMSNorm will be merge into lm_head
        # and this weight will not be merged into the embeddings
        # making the weights of lm_head and embed_tokens not the same
        print("tie word embeddings, clone lm_head from embed_tokens")
        model.config.tie_word_embeddings = False

        # create a new weight for lm_head
        new_weight = torch.empty_like(model.model.embed_tokens.weight)
        new_weight.copy_(model.model.embed_tokens.weight)

        # copy from model.model.embed_tokens.weight
        model.lm_head.weight = nn.Parameter(new_weight)
        new_weight = torch.empty_like(model.model.embed_tokens.weight)
        new_weight.copy_(model.model.embed_tokens.weight)

        # assign the new weight to lm_head
        model.lm_head.weight = nn.Parameter(new_weight)

        # ensure that the ptr of weight of lm_head is not the same as ptr of the weight of embed_tokens
        assert model.model.embed_tokens.weight.data_ptr() != model.lm_head.weight.data_ptr()


@torch.inference_mode()
def rotate_model(model: LlamaForCausalLM,
                 R: torch.Tensor,
                 R_v: list[torch.Tensor] = None):
    config = model.config
    dim = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = dim // num_heads
    num_layers = config.num_hidden_layers

    assert R.shape == (dim, dim), f"Rotation matrix shape {R.shape} does not match model dimension {dim}"

    if isinstance(R_v, torch.Tensor):
        # R_v is a single rotation matrix
        assert R_v.shape == (head_dim, head_dim), f"Rotation matrix shape {R_v.shape} does not match model dimension {dim}"
        R_v = [R_v for _ in range(num_layers)]

    assert R_v is None or len(R_v) == num_layers, f"number of rotation matrix {len(R_v)} does not match number of layers {num_layers}"
    assert all([R_v[i].shape == (head_dim, head_dim) for i in range(num_layers)]) if R_v is not None else True, f"Rotation matrix shape {R_v} does not match model dimension {dim}"

    # rotate embedding
    AutoOperation.rotate_output(model.model.embed_tokens, R)

    for l, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        # reverse rotation for input of W_qkv
        AutoOperation.rotate_input(attn.q_proj, R.T)
        AutoOperation.rotate_input(attn.k_proj, R.T)
        AutoOperation.rotate_input(attn.v_proj, R.T)
        # rotate output of W_o
        AutoOperation.rotate_output(attn.o_proj, R)

        if R_v is not None:
            # rotate v in attention and rotate back before W_o
            AutoOperation.rotate_attn_v(attn, R_v[l])

        mlp = layer.mlp
        # reverse rotation for input of W_up and W_gate
        AutoOperation.rotate_input(mlp.up_proj, R.T)
        AutoOperation.rotate_input(mlp.gate_proj, R.T)
        # rotate output of W_down
        AutoOperation.rotate_output(mlp.down_proj, R)

    # reverse rotation for input of W_lm
    AutoOperation.rotate_input(model.lm_head, R.T)


@RotateOperationRegistry.register(LlamaForCausalLM)
def apply_untie_word_embeddings(model: LlamaForCausalLM, *args, **kwargs):
    """
    Untie the word embeddings of the model.
    """
    print("Untie word embeddings")
    untie_word_embeddings(model)


from ..rotation_utils import fuse_layer_norms

@RotateOperationRegistry.register(LlamaForCausalLM)
def apply_fuse_layer_norms(model: LlamaForCausalLM, *args, **kwargs):
    """
    Fuse the layer norms of the model.
    """
    print("Fuse layer norms")
    fuse_layer_norms(model)


@RotateOperationRegistry.register(LlamaForCausalLM)
def apply_rotate_model(model: LlamaForCausalLM, *args, **kwargs):
    """
    Rotate the model.
    """
    print("Rotate model")
    rotate_model(model, *args, **kwargs)

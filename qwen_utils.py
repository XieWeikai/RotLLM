import torch
from torch import nn
import typing
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from rotation_utils import rotate_linear_input, rotate_linear_output, rotate_embedding, rotate_attn_v

def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_layer_norms(model):
    layers = [layer for layer in model.model.layers]

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
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

    fuse_ln_linear(
        model.model.norm,
        [model.lm_head],
    )
    W_norm = model.model.norm.weight.data
    model.model.norm.weight.data = torch.ones_like(W_norm)
    
    

@torch.inference_mode()
def rotate_model(model: Qwen2ForCausalLM, 
                 R: torch.Tensor,
                 R_v: torch.Tensor = None):
    config = model.config
    dim = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = dim // num_heads
    
    assert R.shape == (dim, dim), f"Rotation matrix shape {R.shape} does not match model dimension {dim}"
    assert R_v is None or R_v.shape == (head_dim, head_dim), f"Rotation matrix shape {R_v.shape} does not match model dimension {num_heads}x{head_dim}"

    # rotate embedding
    rotate_embedding(model.model.embed_tokens, R)
    
    for layer in model.model.layers:
        attn = layer.self_attn
        # reverse rotation for input of W_qkv
        rotate_linear_input([attn.q_proj, attn.k_proj, attn.v_proj], R.T)
        # rotate output of W_o
        rotate_linear_output([attn.o_proj], R)
        
        if R_v is not None:
            # rotate v in attention and rotate back before W_o
            rotate_attn_v(attn, R_v)
        
        mlp = layer.mlp
        # reverse rotation for input of W_up and W_gate
        rotate_linear_input([mlp.up_proj, mlp.gate_proj], R.T)
        # rotate output of W_down
        rotate_linear_output([mlp.down_proj], R)
        
    # reverse rotation for input of W_lm
    rotate_linear_input([model.lm_head], R.T)
    

from torch.nn import init
import rotation_utils

def test_rotate_linear():
    n = 10
    dim = 1536
    
    with torch.no_grad():
        # generate random input and rotation matrix
        x = torch.randn(n, dim)
        R = rotation_utils.get_orthogonal_matrix(dim, "hadamard", "cpu").to(dtype=x.dtype, device=x.device)
        x_rotated = x @ R
    
        ###################################
        # test linear without bias
        linear1 = nn.Linear(dim, dim, bias=False)
        init.xavier_uniform_(linear1.weight)
        linear2 = nn.Linear(dim, dim, bias=False)
        # copy the weight of linear1 to linear2
        linear2.weight.copy_(linear1.weight)
        assert linear1.weight.data.data_ptr() != linear2.weight.data.data_ptr()
        assert torch.allclose(linear1.weight.data, linear2.weight.data, atol=1e-5)

        
        # test rotate_linear_input
        x_ref = linear1(x_rotated)
        rotate_linear_input([linear1], R)
        x_rotated_out = linear1(x)
        assert torch.allclose(x_ref, x_rotated_out, atol=1e-5)
        
        # test rotate_linear_output
        o = linear2(x)
        o_rotated = o @ R
        rotate_linear_output([linear2], R)
        x_out_rotated = linear2(x)
        assert torch.allclose(o_rotated, x_out_rotated, atol=1e-5)
        ##################################
        # test linear with bias
        linear1 = nn.Linear(dim, dim, bias=True)
        init.xavier_uniform_(linear1.weight)
        init.normal_(linear1.bias)
        linear2 = nn.Linear(dim, dim, bias=True)
        
        # copy the weight and bias of linear1 to linear2
        linear2.weight.copy_(linear1.weight)
        linear2.bias.copy_(linear1.bias)
        assert linear1.weight.data.data_ptr() != linear2.weight.data.data_ptr()
        assert linear1.bias.data.data_ptr() != linear2.bias.data.data_ptr()
        assert torch.allclose(linear1.weight.data, linear2.weight.data, atol=1e-5)
        assert torch.allclose(linear1.bias.data, linear2.bias.data, atol=1e-5)
        
        # test rotate_linear_input
        x_ref = linear1(x_rotated)
        rotate_linear_input([linear1], R)
        x_rotated_out = linear1(x)
        assert torch.allclose(x_ref, x_rotated_out, atol=1e-5)
        
        # test rotate_linear_output
        o = linear2(x)
        o_rotated = o @ R
        rotate_linear_output([linear2], R)
        x_out_rotated = linear2(x)
        assert torch.allclose(o_rotated, x_out_rotated, atol=1e-5)
        
        
    return True
        

def test_rotate_embedding():
    n = 10
    dim = 1536
    vocab_size = 150000
    
    with torch.no_grad():
        # generate random input_ids of length n
        input_ids = torch.randint(0, vocab_size, (n,))

        emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        init.xavier_uniform_(emb.weight)
        
        # generate random rotation matrix
        R = rotation_utils.get_orthogonal_matrix(dim, "hadamard", "cpu").to(
            dtype=emb.weight.dtype, device=emb.weight.device
        )
        
        x = emb(input_ids)
        x_ref = x @ R
        
        rotate_embedding(emb, R)
        x_rotated_out = emb(input_ids)
        assert torch.allclose(x_ref, x_rotated_out, atol=1e-5)
        
        
    return True
    
    
def main():
    print(f"test_rotate_embedding: {test_rotate_embedding()}")
    print(f"test_rotate_linear: {test_rotate_linear()}")


if __name__ == "__main__":
    main()
    
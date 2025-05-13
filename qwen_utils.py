import torch
from torch import nn
from typing import Union, Iterable, List
from transformers import Qwen2ForCausalLM, Qwen2VLForConditionalGeneration
from rotation_utils import rotate_linear_input, rotate_linear_output, rotate_embedding, rotate_attn_v
import model_utils


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


class RMSNorm(nn.Module):
    def __init__(self, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"eps={self.variance_epsilon}"


def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: Iterable[torch.nn.Linear]) -> None:
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


def fuse_layer_norms(model: nn.Module, replace_ln: bool = False) -> None:
    it = model_utils.NormLinearIterator.from_model(model)
    
    for father, norm_name, linears in it:
        # fuse the linear operations in Layernorm into the adjacent linear blocks.
        norm = getattr(father, norm_name)
        fuse_ln_linear(norm, linears)
        if not replace_ln: # keep the original layernorm/RMSNorm
            W_norm = norm.weight.data
            norm.weight.data = torch.ones_like(W_norm)
            if hasattr(norm, 'bias'):
                b_norm = norm.bias.data
                norm.bias.data = torch.zeros_like(b_norm)
        else:
            eps = 1e-6
            if hasattr(norm, 'variance_epsilon'):
                eps = norm.variance_epsilon
            if hasattr(norm, 'eps'):
                eps = norm.eps
            # replace the layernorm with RMSNorm
            new_norm = RMSNorm(eps=eps)
            setattr(father, norm_name, new_norm)
    

@torch.inference_mode()
def rotate_model(model: Union[Qwen2ForCausalLM, Qwen2VLForConditionalGeneration], 
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
    rotate_embedding(model.model.embed_tokens, R)
    
    if isinstance(model, Qwen2VLForConditionalGeneration):
        # rotate the output of ViT
        merger = model.visual.merger
        rotate_linear_output([merger.mlp[2]], R)
    
    for l, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        # reverse rotation for input of W_qkv
        rotate_linear_input([attn.q_proj, attn.k_proj, attn.v_proj], R.T)
        # rotate output of W_o
        rotate_linear_output([attn.o_proj], R)
        
        if R_v is not None:
            # rotate v in attention and rotate back before W_o
            rotate_attn_v(attn, R_v[l])
        
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
    
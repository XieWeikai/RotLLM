import typing
import torch
from torch import nn
from hadamard_utils import random_hadamard_matrix

from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.
    
    Args:
    size (int): The size of the matrix (size x size).
    
    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q

def get_orthogonal_matrix(size, mode, device="cpu"):
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f'Unknown mode {mode}')
    

def rotate_linear_input(
    linears: typing.Iterable[torch.nn.Linear],
    R: torch.Tensor):
    """
    Rotate the input of linear layers by a rotation matrix.
    i.e. xW + b -> (xR)W + b ==> x(RW) + b
    This is done by multiplying the weight matrix by the rotation matrix.
    The rotation matrix should be orthogonal.
    """
    for linear in linears:
        dtype = linear.weight.dtype
        R_device = R.device
        w_device = linear.weight.device
        W_ = linear.weight.data.to(device=R_device, dtype=torch.float64)
        # note that the W_ in linear is transpose of W
        linear.weight.data = (W_ @ (R.T.to(torch.float64))).to(device=w_device, dtype=dtype)
        

# denote centering the vector x as C(x) = x - mu
# we have C(x) = x - mu = x - mu 1 where 1 is the vector of ones
#  = x - 1/d sum(x) 1
# we have sum(x) = x_1 + x_2 + ... + x_n = 1^T x
# so we have C(x) = x - 1/d (1^T x) 1 = x - 1/d 1 (1^T x) = x - 1/d 1 1^T x
# that is, we can write C(x) = (I - 1/d 1 1^T) x
# denote the matrix I - 1/d 1 1^T as C
# we have C(x) = C x
# here all the vectors are column vectors
# it is easy to see that C is a symmetric matrix
# so for a row vector x we have C(x) = x C
def center_linear_input(
    linears: typing.Iterable[torch.nn.Linear]):
    """
    Center the input of linear layers
    i.e. xW + b -> (x - mu)W + b
    let C(x) = x - mu = x C
    (x - mu)W + b = x CW + b
    we know that CW can be view as centering the weight matrix by column
    """
    for linear in linears:
        dtype = linear.weight.dtype
        W_ = linear.weight.data.to(dtype=torch.float64)
        # note that the W_ in linear is transpose of W
        # center echo columns of W equivalent to centering the rows of W_
        W_mean = W_.mean(dim=1, keepdim=True)
        W_centered = W_ - W_mean
        linear.weight.data = W_centered.to(dtype=dtype)


def rotate_linear_output(
    linears: typing.Iterable[torch.nn.Linear],
    R: torch.Tensor):
    """
    Rotate the output of linear layers by a rotation matrix.
    i.e. o = xW + b -> o = (xW + b)R ==> o = x(WR) + bR
    This is done by multiplying the weight matrix by the rotation matrix.
    The rotation matrix should be orthogonal.
    """
    for linear in linears:
        dtype = linear.weight.dtype
        R_device = R.device
        w_device = linear.weight.device
        W_ = linear.weight.data.to(device=R_device, dtype=torch.float64)
        # note that the W_ in linear is transpose of W
        linear.weight.data = (R.T.to(torch.float64) @ W_).to(device=w_device, dtype=dtype)
        # rotate the bias
        if linear.bias is not None:
            bias = linear.bias.data.to(device=R_device, dtype=torch.float64)
            linear.bias.data = (bias @ R.to(torch.float64)).to(device=linear.bias.device, 
                                                               dtype=linear.bias.dtype)
    
def center_linear_output(
    linears: typing.Iterable[torch.nn.Linear]):
    """
    Center the output of linear layers
    i.e. xW + b -> (xW + b) C = xW C + bC
    that is we need to center the weight matrix by row and the bias
    """
    for linear in linears:
        dtype = linear.weight.dtype
        W_ = linear.weight.data.to(dtype=torch.float64)
        # note that the W_ in linear is transpose of W
        # center echo columns of W equivalent to centering the rows of W_
        W_mean = W_.mean(dim=0, keepdim=True)
        W_centered = W_ - W_mean
        linear.weight.data = W_centered.to(dtype=dtype)
        if linear.bias is not None:
            bias = linear.bias.data.to(dtype=torch.float64)
            bias_mean = bias.mean()
            bias_centered = bias - bias_mean
            linear.bias.data = bias_centered.to(dtype=dtype)

    
def rotate_embedding(
    embedding: torch.nn.Embedding,
    R: torch.Tensor):
    """
    Rotate each embedding vector by a rotation matrix R.
    """
    dtype = embedding.weight.dtype
    R_device = R.device
    w_device = embedding.weight.device
    W_ = embedding.weight.data.to(device=R_device, dtype=torch.float64)
    # note that the W_ in linear is transpose of W
    embedding.weight.data = (W_ @ (R.to(torch.float64))).to(device=w_device, dtype=dtype)

from typing import Union
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention

def rotate_attn_v(
    attn: Union[Qwen2Attention, Qwen2VLAttention],
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
    rotate_linear_output([attn.v_proj], R_v_rot)
    
    # then we need to rotate back the input of W_o
    # since o_i is linear combination of v_i
    # we can rotate the o_i by R_v^T to get back the original o_i
    rotate_linear_input([attn.o_proj], torch.block_diag(*([R_v] * num_qo_heads)).T)


def test_center_input():
    """
    Test the center_input_output function.
    """
    from torch.nn import init
    
    n = 4
    dim_in = 512
    dim_out = 1024
    x = torch.randn(n, dim_in)
    l = nn.Linear(dim_in, dim_out)
    # initialize the weight matrix
    init.xavier_uniform_(l.weight)
    # randomly initialize the bias
    init.uniform_(l.bias)
    
    # center the input
    x_centered = x - x.mean(dim=1, keepdim=True)
    x_ref = l(x_centered)
    
    # center the weight matrix
    center_linear_input([l])
    x_output = l(x)
    # check if the output is the same
    assert torch.allclose(x_output, x_ref, atol=1e-5), f"Output is not the same: {x_output} != {x_ref}"
    print("Center input test passed.")
    

def test_center_output():
    """
    Test the center_input_output function.
    """
    from torch.nn import init
    
    n = 4
    dim_in = 512
    dim_out = 1024
    x = torch.randn(n, dim_in)
    l = nn.Linear(dim_in, dim_out)
    # initialize the weight matrix
    init.xavier_uniform_(l.weight)
    # randomly initialize the bias
    init.uniform_(l.bias)
    
    # center the output
    x_o = l(x)
    x_ref = x_o - x_o.mean(dim=1, keepdim=True)
    
    # center the weight matrix
    center_linear_output([l])
    x_output = l(x)
    
    # check if the output is the same
    assert torch.allclose(x_output, x_ref, atol=1e-5), f"Output is not the same: {x_output} != {x_ref}"
    print("Center output test passed.")
    

if __name__ == "__main__":
    test_center_input()
    test_center_output()
    

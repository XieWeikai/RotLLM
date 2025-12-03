import torch
import triton
import triton.language as tl

@triton.jit
def quant_per_block_int8_kernel(Input, Scale, L,
                                stride_iz, stride_ih, stride_in,
                                stride_sz, stride_sh,
                                C: tl.constexpr, BLK: tl.constexpr,
                                quant_max: tl.constexpr,  # 127 for int8, 32767 for int16
                                dtype: tl.constexpr):     # tl.int8 or tl.int16
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)

    scale = tl.max(tl.abs(x)) / quant_max
    tl.store(scale_ptrs, scale)

def per_block_int8(input, bits=8, BLKQ=128, BLKK=64, tensor_layout="HND"):
    """
    支持 bits = 8 或 16
    """
    assert bits in [8, 16], "Only support bits=8 or 16"

    quant_max = 127 if bits == 8 else 32767
    dtype = torch.int8 if bits == 8 else torch.int16
    tl_dtype = tl.int8 if bits == 8 else tl.int16

    if tensor_layout == "HND":
        b, h_input, input_len, head_dim = input.shape

        stride_bz_input, stride_h_input, stride_seq_input = input.stride(0), input.stride(1), input.stride(2)
    elif tensor_layout == "NHD":
        b, input_len, h_input, head_dim = input.shape

        stride_bz_input, stride_h_input, stride_seq_input = input.stride(0), input.stride(2), input.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    input_scale = torch.empty((b, h_input, (input_len + BLKQ - 1) // BLKQ, 1), device=input.device, dtype=torch.float32)

    grid = ((input_len + BLKQ - 1) // BLKQ, h_input, b)
    quant_per_block_int8_kernel[grid](
        input, input_scale, input_len,
        stride_bz_input, stride_h_input, stride_seq_input,
        input_scale.stride(0), input_scale.stride(1),
        C=head_dim, BLK=BLKQ,
        quant_max=quant_max,
        dtype=tl_dtype
    )

    return input_scale

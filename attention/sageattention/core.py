import torch
from typing import Any, Optional


from .attn_qkv_int_per_block_causal import forward as attn_true
from .backward import _attn_bwd_preprocess, _attn_bwd


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, scaling, tensor_layout, quant_max):
        dtype = q.dtype
        
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        o = attn_true(q, k, v, M, scaling, quant_max, tensor_layout=tensor_layout, output_dtype=dtype)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.scaling = scaling
        ctx.HEAD_DIM = q.shape[-1]
        ctx.tensor_layout = tensor_layout
        return o
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors

        if ctx.tensor_layout == "HND":
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            o = o.contiguous()
            M = M.contiguous()
            do = do.contiguous()
        elif ctx.tensor_layout == "NHD":
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            o = o.transpose(1, 2).contiguous()
            M = M.transpose(1, 2).contiguous()
            do = do.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Unknown tensor layout: {ctx.tensor_layout}")


        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 2
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.scaling * RCP_LN2)
        PRE_BLOCK = 128
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do,  
            delta,  
            BATCH, N_HEAD, N_CTX,  
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM  
        )
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q, arg_k, v, ctx.scaling, do, dq, dk, dv,  
            M, delta,  
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  
            N_HEAD, N_CTX,  
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  
            HEAD_DIM=ctx.HEAD_DIM,  
            num_warps=NUM_WARPS,  
            num_stages=NUM_STAGES  
        )

        if ctx.tensor_layout == "HND":
            pass
        elif ctx.tensor_layout == "NHD":
            dq = dq.transpose(1, 2).contiguous()
            dk = dk.transpose(1, 2).contiguous()
            dv = dv.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Unknown tensor layout: {ctx.tensor_layout}")

        
        return dq, dk, dv, None, None, None, None


def Quant_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None, 
    is_causal: Optional[bool] = None,
    tensor_layout: str = "HND",
    bits: int = 8,
    **kwargs: Any,
) -> torch.Tensor:
    
    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16, torch.float32], "Input tensors must be in dtype of torch.float16, torch.bfloat16, or torch.float32."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    headdim = q.size(-1)
    assert headdim in [64, 128], "headdim should be in [64, 96, 128]."

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    seq_dim = 1 if tensor_layout == "NHD" else 2


    if scale is None:
        scale = headdim**-0.5
    quant_max =  (2 ** (bits - 1)) - 1
    return _attention.apply(q, k, v, scale, tensor_layout, quant_max)


import torch
from typing import Optional, Any


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """
    Straight-Through Estimator for the round function.
    """
    # 核心实现：前向是 round(x)，后向梯度是 1
    return x + (torch.round(x) - x).detach()


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
    **kwargs: Any
) -> torch.Tensor:
    
    if tensor_layout == "HND":
        pass
    elif tensor_layout == "NHD":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    dtype = q.dtype
    headdim = q.size(-1)
    if scale is None:
        scale = headdim**-0.5
    qk_scale = scale * 1.44269504

    qk = torch.matmul(q, k.transpose(-2, -1))
    
    qk = qk * qk_scale

    qo_len = qk.shape[-2]
    kv_len = qk.shape[-1]
    causal_mask = torch.triu(torch.ones(qo_len, kv_len, device=qk.device, dtype=torch.bool), diagonal=1)
    
    qk.masked_fill_(causal_mask, float('-inf'))

    
    m_i = torch.max(qk, dim=-1, keepdim=True).values  
    
    qk_exp2 = torch.pow(2.0, qk - m_i) 

    quant_max =  (2 ** (bits - 1)) - 1
    p = ste_round(qk_exp2 * quant_max)
    # p = qk_exp2
    
    l_i = torch.sum(p, dim=-1, keepdim=True)
    acc = torch.matmul(p, v) 
    output = acc / l_i
    output = output.to(dtype=dtype)
  
    if tensor_layout == "NHD":
        output = output.transpose(1, 2)

    return output

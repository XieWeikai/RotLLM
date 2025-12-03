import torch, math
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, qk_scale, kv_len,
                    K_ptrs, V_ptrs, stride_kn, stride_vn, 
                    start_m,  
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr, BITS: tl.constexpr,  
                    ):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        K_ptrs += stride_kn * lo
        V_ptrs += stride_vn * lo
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < (kv_len - start_n)   
        k = tl.load(K_ptrs, mask = k_mask)

        qk = tl.dot(q, k).to(tl.float32)
        qk = qk * qk_scale

        mask = k_mask
        if STAGE == 2:
            mask &= offs_m[:, None] >= (start_n + offs_n[None, :])
        qk += tl.where(mask, 0, float('-inf'))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]

        p = tl.math.exp2(qk)

        quant_max = 127 if BITS == 8 else 32767
        tl_dtype = tl.int8 if BITS == 8 else tl.int16
        
        p_int = p * quant_max
        p_int += 0.5 * tl.where(p_int >= 0, 1, -1)
        p_int = p_int.to(tl_dtype)
        p = p_int.to(tl.float32)

        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_ptrs, mask = offs_n[:, None] < (kv_len - start_n))
        
        if tl_dtype == tl.int8:
            acc += (tl.dot(p.to(v.dtype), v).to(tl.float32))
        elif tl_dtype == tl.int16:
            acc += (tl.dot(p.to(v.dtype), v).to(tl.float32))
        else:
            raise ValueError("Only support bits=8 or 16")

        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(Q, K, V, Out, M,  
              stride_qz, stride_qh, stride_qn,
              stride_kz, stride_kh, stride_kn,  
              stride_vz, stride_vh, stride_vn,  
              stride_oz, stride_oh, stride_on,
              stride_mz, stride_mh, stride_mn,
              sm_scale,  
              qo_len, kv_len, H:tl.constexpr, num_kv_groups:tl.constexpr, 
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,  
              BLOCK_N: tl.constexpr, 
              BITS: tl.constexpr,   
              STAGE: tl.constexpr
              ):
    start_m = tl.program_id(0)

    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None] 
   
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]
    M_block_ptr = M + (off_z * stride_mz + off_h * stride_mh) + offs_m * stride_mn
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.44269504
    
    q = tl.load(Q_ptrs, mask = offs_m[:, None] < qo_len)
    
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, qk_scale, kv_len, K_ptrs, V_ptrs, stride_kn, stride_vn,
                                    start_m,  
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  
                                    4 - STAGE, offs_m, offs_n, BITS 
                                    )

    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, qk_scale, kv_len, K_ptrs, V_ptrs, stride_kn, stride_vn,
                                    start_m,  
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  
                                    2, offs_m, offs_n, BITS 
                                    )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < qo_len))

    quant_max = 127 if BITS == 8 else 32767
    tl.store(M_block_ptr, m_i + tl.math.log2(l_i / quant_max), mask = (offs_m < qo_len))

def forward(q, k, v, M, sm_scale, bits=8, tensor_layout="HND", output_dtype=torch.float16):
    BLOCK_M = 128
    BLOCK_N = 64
    stage = 3

    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
        stride_bz_m, stride_h_m, stride_seq_m = M.stride(0), M.stride(1), M.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)
        stride_bz_m, stride_h_m, stride_seq_m = M.stride(0), M.stride(2), M.stride(1)
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")
    
    assert qo_len == kv_len, "qo_len and kv_len must be equal for causal attention"

    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv

    grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b   )
    _attn_fwd[grid](
        q, k, v, o, M,
        stride_bz_q, stride_h_q, stride_seq_q, 
        stride_bz_k, stride_h_k, stride_seq_k,  
        stride_bz_v, stride_h_v, stride_seq_v,  
        stride_bz_o, stride_h_o, stride_seq_o,
        stride_bz_m, stride_h_m, stride_seq_m,
        sm_scale,
        qo_len, kv_len,
        h_qo, num_kv_groups,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,
        BITS = bits,  
        STAGE=stage,  
        num_warps=4 if head_dim == 64 else 8,
        num_stages=4)
    return o
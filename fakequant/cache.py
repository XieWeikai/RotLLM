import torch
from transformers.cache_utils import DynamicCache
from fakequant.linear import quantize_tensor
from rotate import get_orthogonal_matrix

def rotate_quant_func(num_bits: int = 8, symmetric: bool = False, granularity: str = 'per_channel'):
    def quant_func(x: torch.Tensor):
        if getattr(quant_func, 'R', None) is None:
            dim = x.shape[-1]
            R = get_orthogonal_matrix(dim, mode='hadamard', device=x.device)
            quant_func.R = R
        
        R = quant_func.R
        if R.device != x.device:
            R = R.to(x.device)
        
        if R.dtype != x.dtype:
            R = R.to(x.dtype)
        
        x_rotated = x @ R
        x_quantized = quantize_tensor(x_rotated, num_bits=num_bits, symmetric=symmetric, granularity=granularity, axis=-1)
        # Rotate back after quantization
        x_quantized = x_quantized @ R.T
        return x_quantized
    return quant_func


DEFAULT_QUANT_FUNC = rotate_quant_func(num_bits=8, symmetric=False, granularity='per_channel')


class FakeQuantDynamicCache(DynamicCache):
    """
    A dynamic cache that supports fake quantization.
    """
    def __init__(self, *args, quantize: callable = DEFAULT_QUANT_FUNC, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantize = quantize

    def update(self, key_states, value_states, layer_idx, cache_kwargs = None):
        # key_states: (batch_size, num_heads, seq_len, head_dim)
        # value_states: (batch_size, num_heads, seq_len, head_dim)
        key_states = self.quantize(key_states)
        value_states = self.quantize(value_states)
        
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

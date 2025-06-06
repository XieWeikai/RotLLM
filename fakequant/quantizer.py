import torch
from torch import nn
import dataclasses
from typing import Optional, Tuple

@dataclasses.dataclass
class QuantizeConfig:
    """
    Configuration for quantization.
    """
    is_symmetric: bool = True
    num_bits: int = 8
    granularity: str = 'per_tensor' # 'per_tensor' or 'per_channel'
    
    

def compute_n_bits_min_max(config: QuantizeConfig) -> Tuple[int, int]:
    """
    Compute min and max values for quantization.
    """
    min_val = -2 ** (config.num_bits - 1)
    max_val = 2 ** (config.num_bits - 1) - 1
    return min_val, max_val


def compute_qparams(config: QuantizeConfig, input: torch.Tensor)->Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute scale and zero point for quantization.
    input (torch.Tensor): Input tensor of shape [..., d]
    config (QuantizeConfig): Quantization configuration
    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: Scale and zero point
    """
    
    min_val, max_val = compute_n_bits_min_max(config)
    
    if config.granularity == 'per_tensor':
        input = input.flatten()
    
    min_input = input.min(dim=-1, keepdim=True)
    max_input = input.max(dim=-1, keepdim=True)
    
    if config.is_symmetric:
        # For symmetric quantization, we use the maximum absolute value
        max_abs = torch.max(torch.abs(min_input), torch.abs(max_input))
        scales = (max_abs / max_val).to(torch.float32)
        zero_points = None
    else:
        # For asymmetric quantization, we use the full range
        scales = ((max_input - min_input) / (max_val - min_val)).to(torch.float32)
        # Compute zero point: z = q - r/s
        zero_points = min_val - min_input / scales
        zero_points = zero_points.round().to(torch.int32)
    
    return scales, zero_points


class FakeQuantizer(nn.Module):
    def __init__(self, config: QuantizeConfig):
        super().__init__()
        self.config = config
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Quantize input tensor using computed quantization parameters.
        
        Args:
            input (torch.Tensor): Input tensor to quantize
            
        Returns:
            torch.Tensor: Quantized tensor (in floating point format)
        """
        scales, zero_points = compute_qparams(self.config, input)
        min_val, max_val = compute_n_bits_min_max(self.config)
        
        # Reshape scales and zero_points if per-channel quantization
        if self.config.granularity == 'per_channel':
            new_shape = [1] * len(input.shape)
            new_shape[-1] = input.shape[-1]
            scales = scales.view(new_shape)
            if zero_points is not None:
                zero_points = zero_points.view(new_shape)
        
        # Use FakeQuantizeFunction for proper autograd support
        return FakeQuantizeFunction.apply(
            input,
            scales,
            zero_points if not self.config.is_symmetric else torch.zeros_like(scales, dtype=torch.int32),
            min_val,
            max_val
        )

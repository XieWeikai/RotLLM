import torch
import torch.nn as nn


from train.config import WeightQuantizeConfig
from train.quantizer import compute_n_bits_min_max, compute_qparams_dynamic

"""
This class is used to deploy quantized models (int weight + scale) and GPTQ quantization.
"""

class WeightQuantizer(nn.Module):
    def __init__(self, config: WeightQuantizeConfig):
        self.config = config
        self.min_val = None
        self.max_val = None
        self.scale = None
        self.zero_point = None
    
    def find_params(self, input) -> None:
        self.min_val, self.max_val = compute_n_bits_min_max(self.config)
        self.scale, self.zero_point = compute_qparams_dynamic(input, self.config, self.min_val, self.max_val)

    def fake_quantize(self, input):
        if self.zero_point is not None:
            quantized = torch.clamp(torch.round(input / self.scale) + self.zero_point, self.min_val, self.max_val)
            dequantized = (quantized - self.zero_point) * self.scale
        else:
            quantized = torch.clamp(torch.round(input / self.scale), self.min_val, self.max_val)
            dequantized = quantized * self.scale 
        return dequantized, quantized, self.scale 

    def ready(self):
        return self.scale is not None
        
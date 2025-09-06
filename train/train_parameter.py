import torch
import torch.nn as nn


from .config import QuantizeConfig
from .quantizer import (
    compute_n_bits_min_max, 
    compute_qparams_dynamic, 
    compute_qparams_static, 
    StaticLearnableFakeQuantizeFunction, 
    DynamicUnLearnableFakeQuantizeFunction
)

class LearnRotateModule(nn.Module):
    """
    Learnable Rotation matrix: R1, R2
    """
    def __init__(self, R):
        super(LearnRotateModule, self).__init__()
        self.weight = nn.Parameter(R)

class NoLearnRotateModule(nn.Module):
    """
    Unlearnable Rotation matrix: R3, R4
    """
    def __init__(self, R):
        super(NoLearnRotateModule, self).__init__()
        self.weight = R



class FakeQuantizer(nn.Module):
    def __init__(self, config: QuantizeConfig):
        super().__init__()
        self.config = config

    def forward(self, input):
        if self.config.num_bits == 16:      # No quantizer
            return input
        if self.config.mode == 'static':    # Only Support per-tensor and per-channel quantizer
            # If it is static quantization, it is necessary to use the calibration
            # to pre-calculate the scale and zero_point, and set them as learnable parameters.
            if not self.ready():            
                self.qmin, self.qmax = compute_n_bits_min_max(self.config)
                self.scale, self.zero_point = compute_qparams_static(input, self.config, self.qmin, self.qmax)
                self.scale = nn.Parameter(self.scale)
                self.zero_point = nn.Parameter(self.zero_point) if self.zero_point is not None else None
            input_q = StaticLearnableFakeQuantizeFunction.apply(input, self.scale, self.zero_point, self.qmin, self.qmax)
        elif self.config.mode == 'dynamic': # Only Support per-channel and per-group quantizer
            self.qmin, self.qmax = compute_n_bits_min_max(self.config)
            self.scale, self.zero_point = compute_qparams_dynamic(input, self.config, self.qmin, self.qmax)
            input_q = DynamicUnLearnableFakeQuantizeFunction.apply(input, self.scale, self.zero_point, self.qmin, self.qmax)
            
        return input_q

    def ready(self) -> bool :
        if not hasattr(self, "scale"):
            if self.config.mode == 'static':
                return False
            else:
                return True      
        return True
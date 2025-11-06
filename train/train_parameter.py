import torch
import torch.nn as nn


from .config import QuantizeConfig, WeightQuantizeConfig, ActivationQuantizeConfig, KeyQuantizeConfig, ValueQuantizeConfig
from .quantizer import (
    compute_n_bits_min_max, 
    compute_qparams_dynamic, 
    compute_qparams_static_min_max, 
    compute_input_min_max_static,
    compute_qparams_static_mean_std,
    StaticLearnableFakeQuantizeFunction, 
    DynamicUnLearnableFakeQuantizeFunction
)

class LearnRotateModule(nn.Module):
    """
    Learnable Rotation matrix: R1, R2
    """
    def __init__(self, R):
        super(LearnRotateModule, self).__init__()
        self.weight = nn.Parameter(R.to(torch.float32))
    
class NoLearnRotateModule(nn.Module):
    """
    Unlearnable Rotation matrix: R3, R4
    """
    def __init__(self, R):
        super(NoLearnRotateModule, self).__init__()
        self.weight = R.to(torch.float32)



class FakeQuantizer(nn.Module):
    def __init__(self, config: QuantizeConfig):
        super().__init__()
        self.config = config

    def init_activation_scale_and_zero_point_maxmin(self, input):
        # If it is static quantization, it is necessary to use the calibration
        # to pre-calculate the scale and zero_point, and set them as learnable parameters.           
        xmax, xmin = compute_input_min_max_static(input, self.config)
        self.xmax = torch.maximum(self.xmax, xmax) if hasattr(self, "xmax") else xmax
        self.xmin = torch.minimum(self.xmin, xmin) if hasattr(self, "xmin") else xmin
        self.config.need_sample_for_static_init -= 1        # The required sample minus 1

        if self.config.need_sample_for_static_init == 0:
            self.qmin, self.qmax = compute_n_bits_min_max(self.config)
            self.scale, self.zero_point = compute_qparams_static_min_max(self.config, self.xmax, self.xmin, self.qmin, self.qmax)
            self.scale = nn.Parameter(self.scale)
            self.zero_point = nn.Parameter(self.zero_point) if self.zero_point is not None else None

    def init_activation_scale_and_zero_point_mean(self, input):
        self.config.need_sample_for_static_init -= 1        # The required sample minus 1
        self.qmin, self.qmax = compute_n_bits_min_max(self.config)
        scale, zero_point = compute_qparams_static_mean_std(input, self.config, self.qmin, self.qmax)
        if hasattr(self, "scale"):
            self.scale.data = 0.9 * self.scale.data + 0.1 * scale
            if self.zero_point is not None:
                self.zero_point.data = 0.9 * self.zero_point.data + 0.1 * zero_point
        else:
            self.scale = nn.Parameter(scale)
            self.zero_point = nn.Parameter(zero_point) if zero_point is not None else None


    def init_weight_scale_and_zero_point_maxmin(self, input):
        self.config.need_sample_for_static_init -= 1        # The required sample minus 1
        # If it is static quantization, it is necessary to use the calibration
        # to pre-calculate the scale and zero_point, and set them as learnable parameters.  
        if self.ready():
            return         
        xmax, xmin = compute_input_min_max_static(input, self.config)
        self.xmax = xmax
        self.xmin = xmin 
        self.qmin, self.qmax = compute_n_bits_min_max(self.config)
        self.scale, self.zero_point = compute_qparams_static_min_max(self.config, self.xmax, self.xmin, self.qmin, self.qmax)
        self.scale = nn.Parameter(self.scale)
        self.zero_point = nn.Parameter(self.zero_point) if self.zero_point is not None else None

    def init_weight_scale_and_zero_point_mean(self, input):
        self.config.need_sample_for_static_init -= 1        # The required sample minus 1
        if self.ready():
            return
        self.qmin, self.qmax = compute_n_bits_min_max(self.config)
        scale, zero_point = compute_qparams_static_mean_std(input, self.config, self.qmin, self.qmax)
        if hasattr(self, "scale"):
            self.scale.data = 0.9 * self.scale.data + 0.1 * scale
            if self.zero_point is not None:
                self.zero_point.data = 0.9 * self.zero_point.data + 0.1 * zero_point
        else:
            self.scale = nn.Parameter(scale)
            self.zero_point = nn.Parameter(zero_point) if zero_point is not None else None 
        

    def forward(self, input):
        if self.config.num_bits == 16:      # No quantizer
            return input
        if self.config.mode == 'static':    # Only Support per-tensor and per-channel quantizer
            input_q = input
            if isinstance(self.config, (ActivationQuantizeConfig, KeyQuantizeConfig, ValueQuantizeConfig)):
                if self.config.need_sample_for_static_init > 0:
                    if self.config.init_type == 'mean':
                        self.init_activation_scale_and_zero_point_mean(input)
                    elif self.config.init_type == 'maxmin':
                        self.init_activation_scale_and_zero_point_maxmin(input)
                    else:
                        raise NotImplementedError(f"init_type '{self.config.init_type}' is not implemented yet.")
                else:
                    self.qmin, self.qmax = compute_n_bits_min_max(self.config)
                    input_q = StaticLearnableFakeQuantizeFunction.apply(input, self.scale, self.zero_point, self.qmin, self.qmax, self.config.warmup_step, self.config.warmup_share_parameter_num)
            elif isinstance(self.config, WeightQuantizeConfig):
                if self.config.need_sample_for_static_init > 0:         
                    if self.config.init_type == 'mean':
                        self.init_weight_scale_and_zero_point_mean(input)
                    elif self.config.init_type == 'maxmin':
                        self.init_weight_scale_and_zero_point_maxmin(input)
                    else:
                        raise NotImplementedError(f"init_type '{self.config.init_type}' is not implemented yet.")
                else:
                    self.qmin, self.qmax = compute_n_bits_min_max(self.config)
                    input_q = StaticLearnableFakeQuantizeFunction.apply(input, self.scale, self.zero_point, self.qmin, self.qmax)       
        elif self.config.mode == 'dynamic': # Only Support per-channel and per-group quantizer
            input_type = input.dtype
            self.qmin, self.qmax = compute_n_bits_min_max(self.config)
            self.scale, self.zero_point = compute_qparams_dynamic(input.data, self.config, self.qmin, self.qmax)    # 这里使用 .data 避免进入计算图，避免导致 mse 分支的张量占用大量显存不释放
            input_q = DynamicUnLearnableFakeQuantizeFunction.apply(input, self.scale, self.zero_point, self.qmin, self.qmax).to(dtype=input_type)
            
        return input_q

    def ready(self) -> bool :
        if self.config.mode == 'static' and not hasattr(self, "scale"):
            return False   
        return True
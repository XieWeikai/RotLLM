import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Tuple, Optional


def compute_n_bits_min_max(config) -> Tuple[int, int]:
    """
    Compute min and max values for quantization.
    """
    if config.is_symmetric:
        qmin = -(2 ** (config.num_bits - 1))
        qmax =  (2 ** (config.num_bits - 1)) - 1
    else:
        qmin = 0
        qmax = (2 ** config.num_bits) - 1
    return qmin, qmax


def find_params_per_groupwise(input: torch.Tensor, config, min_val, max_val)->Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Dynamic compute scale and zero point for per-group quantization.
    """
    init_shape = input.shape
    reshaped_input = input.reshape(
        -1, input.shape[-2], input.shape[-1] // config.groupsize, config.groupsize
    )

    xmax = torch.amax(reshaped_input, dim=3, keepdim=True) * config.clip_ratio
    xmin = torch.amin(reshaped_input, dim=3, keepdim=True) * config.clip_ratio
    if config.is_symmetric:
        xmax = torch.maximum(torch.abs(xmin), torch.abs(xmax))
        scale = xmax / max_val
        zero_point = None
    else:
        scale = (xmax - xmin) / (max_val - min_val)
        zero_point = torch.round(min_val - xmin / scale)     

    scale = scale.expand(-1, -1, -1, config.groupsize).reshape(init_shape)
    zero_point = zero_point.expand(-1, -1, -1, config.groupsize).reshape(init_shape) if zero_point is not None else None
    return scale, zero_point

def compute_qparams_dynamic(input: torch.Tensor, config, min_val, max_val)->Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Dynamic compute scale and zero point for quantization.
    input (torch.Tensor): Input tensor of shape [..., d]
    config (QuantizeConfig): Quantization configuration
    min_val, max_val: Calculate by compute_n_bits_min_max
    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: Scale and zero point (the shape is the same as input) 
    """
    init_shape = input.shape

    if config.groupsize > 0:
        # group-wise per-token quantization
        scale, zero_point = find_params_per_groupwise(input, config, min_val, max_val)
        return scale, zero_point 

    reshaped_input = input.reshape((-1, input.shape[-1]))
    xmax = torch.amax(reshaped_input, dim=1, keepdim=True) * config.clip_ratio
    xmin = torch.amin(reshaped_input, dim=1, keepdim=True) * config.clip_ratio

    if config.is_symmetric:
        xmax = torch.maximum(torch.abs(xmin), torch.abs(xmax))
        scale = (xmax / max_val).expand(-1, reshaped_input.shape[-1]).reshape(init_shape)
        zero_point = None
    else:
        scale = (xmax - xmin) / (max_val - min_val)
        zero_point = torch.round(min_val - xmin / scale)
        scale = scale.expand(-1, reshaped_input.shape[-1]).reshape(init_shape)
        zero_point = zero_point.expand(-1, reshaped_input.shape[-1]).reshape(init_shape)

    return scale, zero_point 

def compute_qparams_static(input: torch.Tensor, config, min_val, max_val)->Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Static compute scale and zero point for quantization.
    """
    init_shape = input.shape

    if config.granularity == 'per_tensor':
        input = input.flatten() 

    xmax = torch.amax(input, dim=-1, keepdim=True) * config.clip_ratio
    xmin = torch.amin(input, dim=-1, keepdim=True) * config.clip_ratio

    if config.is_symmetric:
        xmax = torch.maximum(torch.abs(xmin), torch.abs(xmax))
        scale = xmax / max_val
        zero_point = None
    else:
        scale = (xmax - xmin) / (max_val - min_val)
        zero_point = torch.round(min_val - xmin / scale)
        
    scale = scale.to(torch.float32)
    zero_point = zero_point.to(torch.float32) if zero_point is not None else None

    return scale, zero_point 


class StaticLearnableFakeQuantizeFunction(torch.autograd.Function):
    """
    Static Quantization(Learnable scale and zero_point):
        1. scale的正约束（使用clamp）
        2. zero_point的整数约束（forward时round）
        3. 梯度裁剪（backward时限制梯度范围）
    """
    @staticmethod
    def forward(ctx, input, scale, zero_point, min_val, max_val):
        # 1. scale正约束
        # constrained_scale = F.softplus(scale) + 1e-6
        # constrained_scale = scale
        constrained_scale = scale.to(input.dtype).clamp(min=1e-6)
        
        # 2. zero_point整数约束（round + clamp）
        if zero_point is not None:
            constrained_z = zero_point.to(input.dtype).round().clamp(min_val, max_val)
        else:
            constrained_z = None
        
        scaled_input = input / constrained_scale
        rounded = scaled_input.round()  

        if constrained_z is not None:
            shifted = rounded + constrained_z
            quantized = torch.clamp(shifted, min_val, max_val)
            dequantized = (quantized - constrained_z) * constrained_scale
        else:
            quantized = torch.clamp(rounded, min_val, max_val)
            dequantized = quantized * constrained_scale
        
        # 保存用于 backward 的参数
        ctx.save_for_backward(input, scale, zero_point)
        ctx.min_val = min_val
        ctx.max_val = max_val
        
        return dequantized

    @staticmethod
    def backward(ctx, grad_output):
        input, scale, zero_point = ctx.saved_tensors
        min_val = ctx.min_val
        max_val = ctx.max_val

        # 重新计算没有保存的中间激活
        # ====================================================================================
        constrained_scale = scale.to(input.dtype).clamp(min=1e-6)
        if zero_point is not None:
            constrained_z = zero_point.to(input.dtype).round().clamp(min_val, max_val)
        else:
            constrained_z = None
        scaled_input = input / constrained_scale
        rounded = scaled_input.round()  

        if constrained_z is not None:
            shifted = rounded + constrained_z
            quantized = torch.clamp(shifted, min_val, max_val)   
        else:
            quantized = torch.clamp(rounded, min_val, max_val)
        # ====================================================================================

        # 1. 输入梯度（STE）
        grad_input = grad_output / constrained_scale
        
        # 2. scale梯度
        if zero_point is not None:
            # mask 对 clamp 的梯度进行截断
            mask = (rounded + constrained_z >= min_val) & (rounded + constrained_z <= max_val)
            term1 = (quantized - constrained_z)
            term2 = (-input / constrained_scale) * mask.float()
            raw_grad_scale = (term1 + term2) * grad_output
        else:
            # mask 对 clamp 的梯度进行截断
            mask = (rounded >= min_val) & (rounded <= max_val)
            term1 = quantized
            term2 = (-input / constrained_scale) * mask.float()
            raw_grad_scale = (term1 + term2) * grad_output
        
        # softplus的导数：d(softplus(x))/dx = sigmoid(x)
        # softplus_deriv = torch.sigmoid(scale)
        # grad_scale = raw_grad_scale * softplus_deriv

        grad_scale = raw_grad_scale
        
        # 3. zero_point梯度（如果存在）
        if zero_point is not None:
            mask = (rounded + constrained_z >= min_val) & (rounded + constrained_z <= max_val)
            raw_grad_z = (mask.float() - 1) * grad_output * constrained_scale
            
            # 由于forward时做了round操作，这里梯度需要特殊处理
            # 使用STE近似：∂round(z)/∂z ≈ 1
            grad_z = raw_grad_z
        else:
            grad_z = None
        
        # 梯度裁剪（防止爆炸）
        max_grad_value = 1.0
        if zero_point is not None:
            grad_z = torch.clamp(grad_z, -max_grad_value, max_grad_value)
        grad_scale = torch.clamp(grad_scale, -max_grad_value, max_grad_value)
        
        # 聚合梯度（保持原始维度）
        if scale.numel() == 1:
            # per-tensor
            grad_scale = grad_scale.sum().unsqueeze(0)
            grad_z = grad_z.sum().unsqueeze(0) if zero_point is not None else None
        else:
            # per-channel: 
            grad_scale = grad_scale.sum(dim=-1, keepdim=True)
            grad_z = grad_z.sum(dim=-1, keepdim=True) if zero_point is not None else None

        return grad_input, grad_scale, grad_z, None, None    


class DynamicUnLearnableFakeQuantizeFunction(torch.autograd.Function):
    """
    Dynamic Quantization(Unlearnable scale and zero_point):
    Custom autograd function for fake quantization with STE (Straight-Through Estimator).
    Performs quantization in forward pass and uses STE for backward propagation.
    """
    @staticmethod
    def forward(ctx, input, scale, zero_point, min_val, max_val):
        """
        Forward pass for fake quantization.
        
        Args:
            ctx: Context object to save tensors for backward pass
            input (Tensor): Input tensor of shape [..., d] to be quantized
            scale (Tensor): Scale tensor of shape [...] (broadcastable to input shape)
            zero_point (Tensor): Zero point tensor (int type) of shape [...] (broadcastable to input shape)
            quant_min (int): Minimum value of quantized integer range
            quant_max (int): Maximum value of quantized integer range
            
        Returns:
            Dequantized tensor with same shape as input
        """  
        if zero_point is not None:
            quantized = torch.clamp(torch.round(input / scale) + zero_point, min_val, max_val)
            dequantized = (quantized - zero_point) * scale
        else:
            quantized = torch.clamp(torch.round(input / scale), min_val, max_val)
            dequantized = quantized * scale 
        return dequantized      

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using Straight-Through Estimator (STE).
        Directly passes gradient through quantization operation.
        
        Args:
            ctx: Context object with saved tensors
            grad_output: Gradient of loss w.r.t. output tensor
            
        Returns:
            Gradients for input
        """       
        # STE: Directly pass gradient through quantization operation
        return grad_output, None, None, None, None

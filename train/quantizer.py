import torch
from typing import Tuple, Optional
import math


from train.config import WeightQuantizeConfig

def quant_dequant(input, scale, zero_point, min_val, max_val):
    if zero_point is not None:
        quantized = torch.clamp(torch.round(input / scale) + zero_point, min_val, max_val)
        dequantized = (quantized - zero_point) * scale
    else:
        quantized = torch.clamp(torch.round(input / scale), min_val, max_val)
        dequantized = quantized * scale 
    return dequantized, quantized, scale  


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
        xmax = torch.maximum(torch.abs(xmin), torch.abs(xmax)).clamp(min=1e-5)
        scale = xmax / max_val
        zero_point = None
    else:
        scale = (xmax - xmin).clamp(min=1e-5) / (max_val - min_val)
        zero_point = torch.round(min_val - xmin / scale)   

    if isinstance(config, WeightQuantizeConfig) and config.mse:
        best_error = torch.full([reshaped_input.shape[0], reshaped_input.shape[1], reshaped_input.shape[2]], float("inf"), device=reshaped_input.device, dtype=reshaped_input.dtype)
        best_scale = scale.clone()
        best_zero = zero_point.clone() if zero_point is not None else None

        for i in range(int(config.grid * config.maxshrink)):
            p = 1 - i / config.grid
            xmin1 = xmin * p
            xmax1 = xmax * p

            if config.is_symmetric:
                scale1 = xmax1 / max_val
                zero_point1 = None
                q, _, _ = quant_dequant(reshaped_input, scale1, zero_point1, min_val, max_val) 
            else:
                scale1 = (xmax1 - xmin1) / (max_val - min_val)
                zero_point1 = torch.round(min_val - xmin1 / scale1)
                q, _, _ = quant_dequant(reshaped_input, scale1, zero_point1, min_val, max_val)  

            err = ((q - reshaped_input).abs() ** config.norm).sum(dim=3)
            mask = err < best_error
            if torch.any(mask):
                best_error[mask] = err[mask]
                best_scale[mask] = scale1[mask]
                if zero_point1 is not None:
                    best_zero[mask] = zero_point1[mask]

        scale = best_scale
        zero_point = best_zero  

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

    tmp = torch.zeros_like(xmax).to(torch.float32)
    xmax = torch.maximum(xmax, tmp)
    xmin = torch.minimum(xmin, tmp)

    if config.is_symmetric:
        xmax = torch.maximum(torch.abs(xmin), torch.abs(xmax)).clamp(min=1e-5)
        scale = xmax / max_val
        zero_point = None
    else:
        scale = (xmax - xmin).clamp(min=1e-5) / (max_val - min_val)
        zero_point = torch.round(min_val - xmin / scale)

    if isinstance(config, WeightQuantizeConfig) and config.mse:
        # best_error = torch.full([reshaped_input.shape[0]], float("inf"), device=reshaped_input.device, dtype=reshaped_input.dtype)
        best_error = torch.full([reshaped_input.shape[0]], float("inf"), device=reshaped_input.device)
        best_scale = scale.clone()
        best_zero = zero_point.clone() if zero_point is not None else None

        for i in range(int(config.grid * config.maxshrink)):
            p = 1 - i / config.grid
            xmin1 = xmin * p
            xmax1 = xmax * p

            if config.is_symmetric:
                scale1 = xmax1 / max_val
                zero_point1 = None
                q, _, _ = quant_dequant(reshaped_input, scale1, zero_point1, min_val, max_val) 
            else:
                scale1 = (xmax1 - xmin1) / (max_val - min_val)
                zero_point1 = torch.round(min_val - xmin1 / scale1)
                q, _, _ = quant_dequant(reshaped_input, scale1, zero_point1, min_val, max_val)  

            err = ((q - reshaped_input).abs() ** config.norm).sum(dim=1)
            mask = err < best_error
            if torch.any(mask):
                best_error[mask] = err[mask]
                best_scale[mask] = scale1[mask]
                if zero_point1 is not None:
                    best_zero[mask] = zero_point1[mask]
              
        scale = best_scale
        zero_point = best_zero

    scale = scale.expand(-1, reshaped_input.shape[-1]).reshape(init_shape)
    zero_point = zero_point.expand(-1, reshaped_input.shape[-1]).reshape(init_shape) if zero_point is not None else None

    scale = scale[..., :1]  
    if zero_point is not None:
        zero_point = zero_point[..., :1]

    return scale, zero_point 

def compute_qparams_static_min_max(config, input_max, input_min, min_val, max_val) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Static compute scale and zero point for quantization.
    """
    if config.is_symmetric:
        input_max = torch.maximum(torch.abs(input_min), torch.abs(input_max)).clamp(min=1e-5)
        scale = input_max / max_val
        zero_point = None
    else:
        scale = (input_max - input_min).clamp(min=1e-5) / (max_val - min_val)
        zero_point = torch.round(min_val - input_min / scale)
        
    scale = scale.to(torch.float32)
    zero_point = zero_point.to(torch.float32) if zero_point is not None else None

    return scale, zero_point 


def compute_input_min_max_static(input: torch.Tensor, config):
    if config.granularity == 'per_tensor':
        input = input.flatten() 

    xmax = torch.amax(input, dim=-1, keepdim=True) * config.clip_ratio
    xmin = torch.amin(input, dim=-1, keepdim=True) * config.clip_ratio

    return xmax, xmin


def compute_qparams_static_mean_std(input: torch.Tensor, config, min_val, max_val):
    if config.granularity == 'per_tensor':
        input = input.flatten() 

    mean = torch.mean(input, dim=-1, keepdim=True)
    std = torch.std(input, dim=-1, keepdim=True)
    xmax = mean + 3 * std
    xmin = mean - 3 * std

    if config.is_symmetric:
        xmax = torch.maximum(torch.abs(xmin), torch.abs(xmax)).clamp(min=1e-5)
        scale = xmax / max_val
        zero_point = None
    else:
        scale = (xmax - xmin).clamp(min=1e-5) / (max_val - min_val)
        zero_point = torch.round(min_val - xmin / scale)
        
    scale = scale.to(torch.float32)
    zero_point = zero_point.to(torch.float32) if zero_point is not None else None

    return scale, zero_point 


class StaticLearnableFakeQuantizeFunction(torch.autograd.Function):
    """
    Static Quantization(Learnable scale and zero_point):
    """
    @staticmethod
    def forward(ctx, input, scale, zero_point, min_val, max_val, warmup_step = None, warmup_share_parameter_num = None):
        input_type = input.dtype
        input = input.to(scale.dtype)

        if zero_point is not None:
            zero_point = zero_point.round()

        # Truncation: In order to accommodate samples with different sequence lengths during evaluation, 
        # it has no effect on the training phase.
        if len(input.shape) >= 3 and len(scale.shape) > 1:
            scale = scale[:, :input.shape[1]]
            if zero_point is not None:
                zero_point = zero_point[:, :input.shape[1]]
        
        scaled_input = input / scale

        if zero_point is not None:
            quantized = torch.clamp(scaled_input + zero_point, min_val, max_val).round()
            dequantized = (quantized - zero_point) * scale
        else:
            quantized = torch.clamp(scaled_input, min_val, max_val).round()
            dequantized = quantized * scale
        
        input = input.to(input_type)
        dequantized = dequantized.to(input_type)
        # Save parameters for backward
        ctx.save_for_backward(input, scale, zero_point)
        ctx.min_val = min_val
        ctx.max_val = max_val
        ctx.warmup_step = warmup_step
        ctx.warmup_share_parameter_num = warmup_share_parameter_num
        
        return dequantized

    @staticmethod
    def backward(ctx, grad_output):
        input, scale, zero_point = ctx.saved_tensors
        min_val = ctx.min_val
        max_val = ctx.max_val
        warmup_step = ctx.warmup_step
        warmup_share_parameter_num = ctx.warmup_share_parameter_num

        input_type = input.dtype
        input = input.to(scale.dtype)

        # Calculate grad_factor
        if scale.numel() == 1:
            # per-tensor
            grad_factor = 1.0 / math.sqrt(input.numel() * max_val)
        else:
            # per-channel: 
            # grad_factor = 1.0 / math.sqrt((input.numel() // input.shape[-1]) * max_val)
            grad_factor = 1.0 / math.sqrt(input.shape[-1] * max_val)

        grad_factor = 1.0
        # 1. Input gradient
        grad_input = grad_output
        
        # 2. Scale gradient
        # 3. Zero_point gradient(if exist)
        scaled_input = input / scale
        if zero_point is not None:
            input_q = scaled_input + zero_point
            quantized = torch.clamp(input_q, min_val, max_val).round()
            dequantized = (quantized - zero_point) * scale
            between = ((input_q > min_val) & (input_q < max_val)).float() 
            smaller = (input_q <= min_val).float() 
            bigger = (input_q >= max_val).float() 

            grad_scale = ((-input_q + quantized) * between + (min_val - zero_point) * smaller + (max_val - zero_point) * bigger) * grad_output * grad_factor
            grad_z = (between - 1) * scale * grad_output * grad_factor
        else:
            input_q = scaled_input
            quantized = torch.clamp(input_q, min_val, max_val).round()
            dequantized = quantized * scale
            between = ((input_q > min_val) & (input_q < max_val)).float() 
            smaller = (input_q <= min_val).float() 
            bigger = (input_q >= max_val).float() 

            grad_scale = ((-input_q + quantized) * between + min_val * smaller + max_val * bigger) * grad_output * grad_factor
            grad_z = None
        
        if warmup_step is not None and warmup_step > 0:
            warmup_step -= 1
            init_grad_scale, init_grad_z = update_activation_init_scale_and_zero_point(input, scale, zero_point, min_val, max_val, grad_factor, warmup_share_parameter_num)
            grad_scale = grad_scale + init_grad_scale
            grad_z = grad_z + init_grad_z if zero_point is not None else None


        # Aggregate gradients (maintain original dimensions)
        if scale.numel() == 1:
            # per-tensor
            grad_scale = grad_scale.sum().unsqueeze(0)
            grad_z = grad_z.sum().unsqueeze(0) if zero_point is not None else None
        else:
            # per-channel: 
            grad_scale = grad_scale.sum(dim=-1, keepdim=True)
            grad_z = grad_z.sum(dim=-1, keepdim=True) if zero_point is not None else None

        assert torch.isfinite(grad_scale).all(), "grad_scale has NaN or Inf"
        assert torch.isfinite(grad_input).all(), "grad_input has NaN or Inf"

        grad_scale = grad_scale.clamp(min=-0.1, max=0.1)
        grad_z = grad_z.clamp(min=-0.1, max=0.1) if grad_z is not None else None

        input = input.to(input_type)
        return grad_input, grad_scale, grad_z, None, None, None, None    

    


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



def update_activation_init_scale_and_zero_point(input, scale, zero_point, min_val, max_val, grad_factor, warmup_share_parameter_num):
    scaled_input = input / scale
    if zero_point is not None:
        input_q = scaled_input + zero_point
        quantized = torch.clamp(input_q, min_val, max_val).round()
        dequantized = (quantized - zero_point) * scale
        between = ((input_q > min_val) & (input_q < max_val)).float() 
        smaller = (input_q <= min_val).float() 
        bigger = (input_q >= max_val).float() 

        # grad_scale = gradient(dequantized / scale)
        grad_scale = ((-input_q + quantized) * between + (min_val - zero_point) * smaller + (max_val - zero_point) * bigger) * grad_factor
        grad_z = (between - 1) * scale * grad_factor
    else:
        input_q = scaled_input
        quantized = torch.clamp(input_q, min_val, max_val).round()
        dequantized = quantized * scale
        between = ((input_q > min_val) & (input_q < max_val)).float() 
        smaller = (input_q <= min_val).float() 
        bigger = (input_q >= max_val).float() 

        grad_scale = ((-input_q + quantized) * between + min_val * smaller + max_val * bigger) * grad_factor
        grad_z = None

    if zero_point is not None:
        grad_scale = 2 * (dequantized - input) * grad_scale / warmup_share_parameter_num
        grad_z = 2 * (dequantized - input) * grad_z / warmup_share_parameter_num
    else:
        grad_scale = 2 * (dequantized - input) * grad_scale / warmup_share_parameter_num
        grad_z = None
    
    return grad_scale, grad_z





class DynamicUnLearnableQKVFakeQuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, min_val, max_val):
        """
        x_int:  [B, 2048, S, 64]
        scale:  [B, 16, S, 1]   (16 blocks)
        """
        B, N, S, D = input.shape
        _, n_blocks, _, _ = scale.shape # 2048/128 = 16
        block_size = N // n_blocks

        # reshape input into blocks
        input = input.view(B, n_blocks, block_size, S, D)  # [4,32,16,128,64]

        # expand scale for broadcasting
        scale = scale.unsqueeze(2)  # [4,32,16,1,1]
        if zero_point is not None:
            zero_point = zero_point.unsqueeze(2)

        if zero_point is not None:
            quantized = torch.clamp(torch.round(input / scale) + zero_point, min_val, max_val)
            dequantized = (quantized - zero_point) * scale
        else:
            quantized = torch.clamp(torch.round(input / scale), min_val, max_val)
            dequantized = quantized * scale

        dequantized = dequantized.view(B, N, S, D)
        return dequantized

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None

import torch
import torch.nn.functional as F


class FakeQuantizeFunction(torch.autograd.Function):
    """
    Custom autograd function for fake quantization with STE (Straight-Through Estimator).
    Performs quantization in forward pass and uses STE for backward propagation.
    """
     
    @staticmethod
    def forward(ctx, input, scales, zero_points, quant_min, quant_max):
        """
        Forward pass for fake quantization.
        
        Args:
            ctx: Context object to save tensors for backward pass
            input (Tensor): Input tensor of shape [..., d] to be quantized
            scales (Tensor): Scale tensor of shape [...] (broadcastable to input shape)
            zero_points (Tensor): Zero point tensor (int type) of shape [...] (broadcastable to input shape)
            quant_min (int): Minimum value of quantized integer range
            quant_max (int): Maximum value of quantized integer range
            
        Returns:
            Dequantized tensor with same shape as input
        """
        
        # Quantization operation
        # 1. Scale input and shift by zero point
        scaled_input = input / scales
        scaled_input = scaled_input.round() # Round to nearest integer
        shifted_input = scaled_input + zero_points
        
        # 2. Clamp to quantization range and round
        quantized = shifted_input
        # outlier mask (in the position of outliers the value is 0)
        # and the gradient is 0
        mask = torch.logical_and(
            quantized >= quant_min,
            quantized <= quant_max,
        )
        ctx.save_for_backward(mask)
        quantized = torch.clamp(quantized, quant_min, quant_max)
        
        # 3. Dequantize back to original scale
        dequantized = (quantized - zero_points) * scales
        
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
        # Retrieve saved tensors from forward pass
        (mask, ) = ctx.saved_tensors
        
        # STE: Directly pass gradient through quantization operation
        return grad_output * mask, None, None, None, None
    
    
class LearnableFakeQuantizeFunction(torch.autograd.Function):
    """
    完整约束条件的可学习量化实现：
    1. scale的正约束（使用softplus）
    2. zero_point的整数约束（forward时round）
    3. 梯度裁剪（backward时限制梯度范围）
    """
    
    @staticmethod
    def forward(ctx, input, scale, zero_point, quant_min, quant_max):
        # 1. scale正约束（使用softplus + 最小值保护）
        constrained_scale = F.softplus(scale) + 1e-6
        
        # 2. zero_point整数约束（round + clamp）
        if zero_point is not None:
            constrained_z = zero_point.round().clamp(quant_min, quant_max)
        else:
            constrained_z = None
        
        # 量化操作
        scaled_input = input / constrained_scale
        rounded = scaled_input.round()  # round(x/s)
        
        if constrained_z is not None:
            shifted = rounded + constrained_z
            quantized = torch.clamp(shifted, quant_min, quant_max)
            dequantized = (quantized - constrained_z) * constrained_scale
        else:
            quantized = torch.clamp(rounded, quant_min, quant_max)
            dequantized = quantized * constrained_scale
        
        # 保存用于backward的参数（保存原始参数而非约束后的）
        ctx.save_for_backward(input, scale, zero_point, rounded, quantized)
        ctx.constrained_scale = constrained_scale
        ctx.constrained_z = constrained_z
        ctx.quant_min = quant_min
        ctx.quant_max = quant_max
        
        return dequantized

    @staticmethod
    def backward(ctx, grad_output):
        input, scale, zero_point, rounded, quantized = ctx.saved_tensors
        constrained_scale = ctx.constrained_scale
        constrained_z = ctx.constrained_z
        quant_min = ctx.quant_min
        quant_max = ctx.quant_max
        
        # 1. 输入梯度（STE）
        grad_input = grad_output / constrained_scale
        
        # 2. scale梯度（考虑softplus的导数）
        if zero_point is not None:
            term1 = (quantized - constrained_z)
            term2 = -input / constrained_scale
            raw_grad_scale = (term1 + term2) * grad_output
        else:
            term1 = quantized
            term2 = -input / constrained_scale
            raw_grad_scale = (term1 + term2) * grad_output
        
        # softplus的导数：d(softplus(x))/dx = sigmoid(x)
        softplus_deriv = torch.sigmoid(scale)
        grad_scale = raw_grad_scale * softplus_deriv
        
        # 3. zero_point梯度（如果存在）
        if zero_point is not None:
            mask = (rounded + constrained_z >= quant_min) & (rounded + constrained_z <= quant_max)
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
        grad_scale = grad_scale.sum(dim=tuple(range(grad_scale.dim()-scale.dim())))
        if zero_point is not None:
            grad_z = grad_z.sum(dim=tuple(range(grad_z.dim()-zero_point.dim())))
        
        return grad_input, grad_scale, grad_z, None, None

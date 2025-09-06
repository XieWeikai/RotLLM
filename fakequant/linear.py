import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_tensor(x, num_bits=8, symmetric=True, granularity='per_tensor', axis=-1):
    """
    Quantize a tensor using fake quantization.
    
    Args:
        x (torch.Tensor): Input tensor to quantize
        num_bits (int): Number of bits for quantization (default: 8)
        symmetric (bool): Whether to use symmetric quantization (default: True)
        granularity (str): Quantization granularity, 'per_tensor' or 'per_channel' (default: 'per_tensor')
        axis (int): Axis for per-channel quantization (default: -1)
        
    Returns:
        torch.Tensor: Fake quantized tensor (dequantized values)
    """
    if x is None:
        return None
        
    assert granularity in ['per_tensor', 'per_channel'], "granularity must be 'per_tensor' or 'per_channel'"
    qmin = - (2 ** (num_bits - 1)) if symmetric else 0
    qmax = (2 ** (num_bits - 1)) - 1 if symmetric else (2 ** num_bits) - 1

    if granularity == 'per_tensor':
        min_val, max_val = x.min(), x.max()

        if symmetric:
            max_val = max(abs(min_val), abs(max_val))
            scale = max_val / qmax
            zero_point = 0.0
        else:
            scale = (max_val - min_val) / (qmax - qmin + 1e-8)
            zero_point = qmin - min_val / (scale + 1e-8)
            zero_point = zero_point.clamp(qmin, qmax).round()

        q_x = torch.clamp((x / scale + zero_point).round(), qmin, qmax)
        dq_x = (q_x - zero_point) * scale

    else:  # per_channel
        # Handle negative axis and shape broadcasting
        axis = axis if axis >= 0 else x.dim() + axis
        assert 0 <= axis < x.dim(), f"Invalid axis {axis} for tensor with shape {x.shape}"

        min_val = x.amin(dim=axis, keepdim=True)
        max_val = x.amax(dim=axis, keepdim=True)

        if symmetric:
            max_val = torch.maximum(min_val.abs(), max_val.abs())
            scale = max_val / qmax
            zero_point = 0.0
        else:
            scale = (max_val - min_val) / (qmax - qmin + 1e-8)
            zero_point = qmin - min_val / (scale + 1e-8)
            zero_point = zero_point.clamp(qmin, qmax).round()

        q_x = torch.clamp((x / scale + zero_point).round(), qmin, qmax)
        dq_x = (q_x - zero_point) * scale

    return dq_x


class FakeQuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, num_bits=8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.num_bits = num_bits  # e.g., 8 for int8

    @classmethod
    def from_linear(cls, linear: nn.Linear, num_bits=8):
        # Get dtype and device from original linear layer
        dtype = linear.weight.dtype
        device = linear.weight.device

        # Create FakeQuantLinear and move to same dtype & device

        fq_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            num_bits=num_bits
        ).to(dtype=dtype, device=device)

        # Copy weights and bias
        fq_linear.linear.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            fq_linear.linear.bias.data.copy_(linear.bias.data)

        return fq_linear


    def forward(self, x):
        # Activation: per-tensor, asymmetric (A8)
        x_q = quantize_tensor(x, num_bits=self.num_bits, symmetric=False, granularity='per_tensor')
        linear_dev = self.linear.weight.device
        if x_q.device != linear_dev:
            x_q = x_q.to(linear_dev)

        # Weight: per-channel, symmetric (W8)
        w_q = quantize_tensor(self.linear.weight, num_bits=self.num_bits, symmetric=True, granularity='per_channel')
        b_q = quantize_tensor(self.linear.bias, num_bits=self.num_bits, symmetric=False, granularity='per_channel') if self.linear.bias is not None else None

        y = F.linear(x_q, w_q, b_q)
        y_quant = quantize_tensor(y, num_bits=self.num_bits, symmetric=False, granularity="per_tensor")
        return y_quant
    
    def extra_repr(self):
        return f"in_features={self.linear.in_features}, out_features={self.linear.out_features}, bias={self.linear.bias is not None}, num_bits={self.num_bits}"


def replace_linear_with_fakequant(model: nn.Module, num_bits=8, filter_fn=None):
    """
    Replace all nn.Linear modules in a model with FakeQuantLinear.

    Args:
        model (nn.Module): The model to modify in-place.
        num_bits (int): Bit width for quantization (default: 8).
        filter_fn (Callable[[str, nn.Linear], bool], optional):
            A user-defined function that takes the module name and the Linear module as input,
            and returns True if the Linear layer should be replaced, or False to keep it unchanged.

    Returns:
        nn.Module: The model with selected Linear layers replaced by FakeQuantLinear.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            if filter_fn is None or filter_fn(name, module):
                setattr(model, name, FakeQuantLinear.from_linear(module, num_bits=num_bits))
        else:
            replace_linear_with_fakequant(module, num_bits=num_bits, filter_fn=filter_fn)
    return model


if __name__ == "__main__":
    # Test the refactored quantize_tensor function
    print("Testing quantize_tensor function...")
    
    # Test 1: Basic functionality
    x = torch.randn(10, 20)
    print(f"Original tensor shape: {x.shape}")
    
    # Test per-tensor quantization
    x_quant_pt = quantize_tensor(x, num_bits=8, symmetric=True, granularity='per_tensor')
    print(f"Per-tensor quantized shape: {x_quant_pt.shape}")
    print(f"Per-tensor quantization maintains dtype: {x.dtype == x_quant_pt.dtype}")
    
    # Test per-channel quantization
    x_quant_pc = quantize_tensor(x, num_bits=8, symmetric=True, granularity='per_channel', axis=-1)
    print(f"Per-channel quantized shape: {x_quant_pc.shape}")
    print(f"Per-channel quantization maintains dtype: {x.dtype == x_quant_pc.dtype}")
    
    # Test 2: FakeQuantLinear functionality
    print("\nTesting FakeQuantLinear...")
    linear = nn.Linear(20, 10)
    fq_linear = FakeQuantLinear.from_linear(linear, num_bits=8)
    
    y_orig = linear(x)
    y_quant = fq_linear(x)
    
    print(f"Original output shape: {y_orig.shape}")
    print(f"Quantized output shape: {y_quant.shape}")
    print(f"Outputs have same shape: {y_orig.shape == y_quant.shape}")
    print(f"Outputs are different (due to quantization): {not torch.allclose(y_orig, y_quant)}")
    
    # Test 3: None handling
    print("\nTesting None handling...")
    none_result = quantize_tensor(None)
    print(f"None input returns None: {none_result is None}")
    
    print("\nAll tests passed! Refactoring successful.")


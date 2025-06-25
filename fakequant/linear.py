import torch
import torch.nn as nn
import torch.nn.functional as F

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


    def quantize_tensor(self, x, num_bits=8, symmetric=True, granularity='per_tensor', axis=-1):
        assert granularity in ['per_tensor', 'per_channel'], "granularity must be 'per_tensor' or 'per_channel'"
        
        qmin = None if symmetric else 0 # FIXME: calculate qmin based on num_bits
        qmax = None if symmetric else (2 ** num_bits) - 1 # FIXME: calculate qmax based on num_bits

        if granularity == 'per_tensor':
            if symmetric:
                # TODO: calculate scale and zero_point for symmetric quantization
                scale = None
                zero_point = None
            else:
                # TODO: calculate scale and zero_point for asymmetric quantization
                scale = None
                zero_point = None

            q_x = torch.clamp((x / scale + zero_point).round(), qmin, qmax)
            dq_x = (q_x - zero_point) * scale

        else:  # per_channel
            # TODO: quantize along the specified axis
            scale = None
            zero_point = None

            q_x = torch.clamp((x / scale + zero_point).round(), qmin, qmax)
            dq_x = (q_x - zero_point) * scale

        return dq_x


    def forward(self, x):
        # Activation: per-tensor, asymmetric (A8)
        x_q = self.quantize_tensor(x, num_bits=self.num_bits, symmetric=False, granularity='per_tensor')

        # Weight: per-channel, symmetric (W8)
        w_q = self.quantize_tensor(self.linear.weight, num_bits=self.num_bits, symmetric=True, granularity='per_tensor')
        b_q = self.quantize_tensor(self.linear.bias, num_bits=self.num_bits, symmetric=False, granularity='per_tensor') if self.linear.bias is not None else None

        y = F.linear(x_q, w_q, b_q)
        y_quant = self.quantize_tensor(y, num_bits=self.num_bits, symmetric=False, granularity="per_tensor")
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


import torch
import torch.nn as nn
from torch.ao.quantization import FakeQuantize, MinMaxObserver


class ActivationQDQ(nn.Module):
    """
    General activation value pseudo-quantization module (QDQ).
    Supports symmetric Per-Tensor quantization, configurable bit numbers (e.g., 8-bit or 16-bit).
    """

    def __init__(self, bits=8, qscheme=torch.per_tensor_affine):
        super().__init__()

        self.bits = bits
        self.qscheme = qscheme

        # Define the simulation dtype as qint32 to avoid overflow across different bit-widths
        self.dtype = torch.qint32

        # 1. Calculate quantization range based on bits and scheme
        if qscheme in [torch.per_tensor_symmetric, torch.per_channel_symmetric]:
            # Symmetric: range is [-(2^(bits-1)), 2^(bits-1) - 1]
            # e.g., 8-bit: -128 to 127
            self.quant_min = -(2 ** (bits - 1))
            self.quant_max = 2 ** (bits - 1) - 1
        else:
            # Asymmetric (Affine): range is [0, 2^bits - 1]
            # e.g., 8-bit: 0 to 255
            self.quant_min = 0
            self.quant_max = (2**bits) - 1

        # 2. Initialize FakeQuantize
        # MinMaxObserver calculates scale and zero_point based on observed tensors.
        # Passing quant_min/max to the observer ensures consistency.
        self.fake_quant = FakeQuantize(
            observer=MinMaxObserver.with_args(
                qscheme=self.qscheme,
                dtype=self.dtype,
                quant_min=self.quant_min,
                quant_max=self.quant_max,
                reduce_range=False,
            ),
            quant_min=self.quant_min,
            quant_max=self.quant_max,
            dtype=self.dtype,
            qscheme=self.qscheme,
        )

    def forward(self, x):
        # Directly apply pseudo-quantization.
        # When observer is enabled, it continuously updates scale/zp;
        # When fakequant is enabled, it simulates quantization errors.
        return self.fake_quant(x)

    def enable_observer(self):
        self.fake_quant.enable_observer()

    def disable_observer(self):
        self.fake_quant.disable_observer()

    def enable_fakequant(self):
        self.fake_quant.enable_fake_quant()

    def disable_fakequant(self):
        self.fake_quant.disable_fake_quant()

    def extra_repr(self):
        return f"bits={self.quant_max.bit_length() + 1}, q_range=({self.quant_min}, {self.quant_max})"
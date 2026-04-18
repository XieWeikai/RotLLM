import torch.nn as nn
from tqdm import tqdm
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP
from utils.utils import get_text_tower


class Qwen3VLTextMLPWithR4(nn.Module):
    def __init__(self, module: Qwen3VLTextMLP, R4):
        super().__init__()
        self.hidden_size = module.hidden_size
        self.intermediate_size = module.intermediate_size
        self.gate_proj = module.gate_proj
        self.up_proj   = module.up_proj
        self.down_proj = module.down_proj
        self.act_fn    = module.act_fn
        self.R4 = R4
    
    def forward(self, x):
        # We modify (add R4)
        gated_activation = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        gated_activation_dtype = gated_activation.dtype
        gated_activation_device = gated_activation.device
        down_proj = self.down_proj((gated_activation.to(dtype = self.R4.weight.dtype) @ self.R4.weight.to(gated_activation_device)).to(dtype = gated_activation_dtype))
        
        return down_proj
    

def apply_R4_change_model(model, R4_list, local_rank=None):
    """
        Replace Qwen3VLTextMLP with Qwen3VLTextMLPWithR4
    """
    if local_rank is None:
        local_rank = 0

    _, text_model = get_text_tower(model)
    layers = text_model.layers
    
    for i in tqdm(range(len(layers)), desc="Replace Qwen3VLTextMLP with Qwen3VLTextMLPWithR4", disable=not (local_rank == 0)):
        layer = layers[i]
        for name, module in layer.named_children():
            if isinstance(module, Qwen3VLTextMLP):
                # Take out the R4 of the corresponding layer from the list.
                R4 = R4_list[i]
                setattr(layer, name, Qwen3VLTextMLPWithR4(module, R4))
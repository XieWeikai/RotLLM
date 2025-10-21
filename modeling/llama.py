import torch.nn as nn
from tqdm import tqdm
from transformers.models.llama.modeling_llama import LlamaMLP


class LlamaMLPWithR4(nn.Module):
    def __init__(self, module: LlamaMLP, R4):
        super().__init__()
        self.config = module.config
        self.hidden_size = module.hidden_size
        self.intermediate_size = module.intermediate_size
        self.gate_proj = module.gate_proj
        self.up_proj = module.up_proj
        self.down_proj = module.down_proj
        self.act_fn = module.act_fn
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
        Replace LlamaMLP with LlamaMLPWithR4
    """
    if local_rank is None:
        local_rank = 0
    layers = model.model.layers
    for i in tqdm(range(len(layers)), desc="Replace LlamaMLP with LlamaMLPWithR4", disable=not (local_rank == 0)):
        layer = layers[i]
        for name, module in layer.named_children():
            if isinstance(module, LlamaMLP):
                # Take out the R4 of the corresponding layer from the list.
                R4 = R4_list[i]
                setattr(layer, name, LlamaMLPWithR4(module, R4))

import torch
import torch.nn as nn
from tqdm import tqdm
import copy


from .weight_quant import WeightQuantizer
from train.config import QuantizeConfig
from utils.utils import cleanup_memory


def find_qlayers(module, layers=[nn.Linear, nn.Embedding], name: str = ""):
    if type(module) in [nn.Embedding] and type(module) in layers:
        return {"embed_tokens": module}
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_qlayers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res

@torch.no_grad()
def rtn_fwrd(model, wConfig: QuantizeConfig):
    layers = model.model.layers
    quantizers = {}

    for i in tqdm(range(len(layers)), desc="(RtN Quant.) Layers"):
        layer = layers[i]

        subset = find_qlayers(layer, layers=[nn.Linear, nn.Embedding])

        for name in subset:
            wQuantConfig = copy.deepcopy(wConfig)

            if "lm_head" in name:
                wQuantConfig.num_bits = 16
            if wQuantConfig.int8_down_proj and "down_proj" in name:
                wQuantConfig.num_bits = 16

            quantizer = WeightQuantizer(wQuantConfig)
            W = subset[name].weight.data
            quantizer.find_params(W)
            q, int_weight, scale = quantizer.fake_quantize(W)
            subset[name].weight.data = q.to(next(iter(layer.parameters())).dtype)
            quantizers["model.layers.%d.%s" % (i, name)] = quantizer

    cleanup_memory(verbos=True)
    return quantizers
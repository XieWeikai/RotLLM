import torch
from tqdm import tqdm


from train.train_parameter import FakeQuantizer
from train.train_model import RotationEmbedding, RotationQuantLinear
from utils.utils import log




def find_qlayers(module, layers=[RotationQuantLinear, RotationEmbedding], name: str = ""):
    if type(module) in [RotationEmbedding] and type(module) in layers:
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


def static_rtn_fwrd(model):
    """
    遍历 model，找到所有 weightQuant 对象
    
    Args:
        model (nn.Module): 待遍历模型
    """
    

    layers = model.model.layers
    for i in tqdm(range(len(layers)), desc="(Static RtN Quant.) Layers"):
        layer = layers[i]
        subset = find_qlayers(layer, layers=[RotationQuantLinear])
        for name in subset:
            module = subset[name]
            assert module.weightQuant.config.mode == "static", "We are using dynamic quantization!"
            
            W = module.linear.weight.data
            W_type = W.dtype
            module.linear.weight.data = module.weightQuant(W).to(dtype=W_type)
            module.weightQuant.config.num_bits = 32
            if torch.any(torch.isnan(module.linear.weight.data)):
                raise ValueError("NaN in linear weights")
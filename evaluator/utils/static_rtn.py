import torch
from tqdm import tqdm
import torch.nn as nn


from train.train_parameter import FakeQuantizer
from train.prepare_model import model_down_proj_groupsize
from train.train_model import RotationEmbedding, RotationQuantLinear


def change_config_for_static_quant(model, ptq_args):
    num_heads = model.config.num_attention_heads
    dim = model.config.hidden_size
    head_dim = dim // num_heads

    for name, module in model.named_modules():
        if isinstance(module, FakeQuantizer):
            module.config.mode = "static"
            # weight:
            if ptq_args.w_bits < 16 and "weightQuant" in name:
                if "lm_head" in name:
                    module.config.num_bits = 16
                if ptq_args.int8_down_proj and "down_proj" in name:
                    module.config.num_bits = 8

            # activation:
            if ptq_args.a_bits < 16 and "actQuant" in name:
                if "lm_head" in name:
                    module.config.num_bits = 16
                if "down_proj" in name:
                    if ptq_args.int8_down_proj:
                        module.config.num_bits = 8
                    if ptq_args.a_groupsize > 0:
                        down_proj_groupsize = model_down_proj_groupsize(model, ptq_args.a_groupsize)
                        module.config.groupsize = down_proj_groupsize
                if "o_proj" in name:
                    module.config.groupsize = head_dim

            # value:
            if ptq_args.v_bits < 16 and "vQuant" in name:
                if "v_proj" in name:
                    module.config.groupsize = head_dim 


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


def static_rtn_fwrd(model, batch: torch.Tensor, ptq_args):
    """
    遍历 model，找到所有 weightQuant 对象
    
    Args:
        model (nn.Module): 待遍历模型
    """
    change_config_for_static_quant(model, ptq_args)

    model.eval()
    with torch.no_grad(): 
        print("bs:", batch.size(0))
        for i in tqdm(range(batch.size(0)), desc="Init scale and zero_point for static quant"):
            sample = batch[i].unsqueeze(0)  # 保持 batch 维度
            model(sample)
        print("Init scale and zero_point ok!")
    

    layers = model.model.layers
    for i in tqdm(range(len(layers)), desc="(Static RtN Quant.) Layers"):
        layer = layers[i]
        subset = find_qlayers(layer, layers=[RotationQuantLinear, RotationEmbedding])

        for name in subset:
            module = subset[name]
            assert module.weightQuant.config.mode == "static", "We are using dynamic quantization!"
            if isinstance(module, RotationQuantLinear):
                W = module.linear.weight.data
                W_type = W.dtype
                module.linear.weight.data = module.weightQuant(W).to(dtype=W_type)
                module.weightQuant.config.num_bits = 16
                if torch.any(torch.isnan(module.linear.weight.data)):
                    raise ValueError("NaN in linear weights")
            if isinstance(module, RotationEmbedding):
                W = module.embedding.weight.data
                W_type = W.dtype
                module.embedding.weight.data = module.weightQuant(W).to(dtype=W_type)
                module.weightQuant.config.num_bits = 16
                if torch.any(torch.isnan(module.embedding.weight.data)):
                    raise ValueError("NaN in embedding weights")
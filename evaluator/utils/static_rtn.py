import torch
from tqdm import tqdm


from train.train_parameter import FakeQuantizer
from train.train_model import RotationEmbedding, RotationQuantLinear
from utils.utils import log


def change_config_for_static_quant(model, ptq_args):
    num_heads = model.config.num_attention_heads
    dim = model.config.hidden_size
    head_dim = dim // num_heads

    for name, module in model.named_modules():
        if isinstance(module, FakeQuantizer):
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
    # batch_test = batch[-4:]
    # x12 = batch[:12]
    # x_last4 = x12[-4:]
    # batch = torch.cat([x12, x_last4], dim=0)
    # print(batch.shape[0])
    # print(batch_test.shape[0])
    """
    遍历 model，找到所有 weightQuant 对象
    
    Args:
        model (nn.Module): 待遍历模型
    """
    change_config_for_static_quant(model, ptq_args)

    model.eval()
    with torch.no_grad(): 
        for i in tqdm(range(batch.size(0)), desc="Init scale and zero_point for static quant"):
            sample = batch[i].unsqueeze(0)  # 保持 batch 维度
            model(sample)
        log.info("✅ Init scale and zero_point ok!")
    
    # draw_weight(model, batch_test)
    # assert False, "haha"            

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
            module.weightQuant.config.num_bits = 16
            if torch.any(torch.isnan(module.linear.weight.data)):
                raise ValueError("NaN in linear weights")
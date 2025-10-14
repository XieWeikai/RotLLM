import torch
from tqdm import tqdm


from train.train_parameter import FakeQuantizer
from train.train_model import RotationEmbedding, RotationQuantLinear


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


def trainable_static_rtn_fwrd(model, ptq_args, model_args):
    """
    遍历 model，找到所有 FakeQuantizer 对象
    
    Args:
        model (nn.Module): 待遍历模型
    """
    change_config_for_static_quant(model, ptq_args)

    model.eval() 

    layers = model.model.layers
    path = model_args.output_rotation_path
    data = torch.load(path, map_location="cpu")

    # with open("no_quant.txt", "a", encoding="utf-8") as f:
    subset = find_qlayers(model, layers=[FakeQuantizer])
    for name in tqdm(subset, desc="(Trainable Static RtN Quant.) FakeQuantizer"):
        module = subset[name]
        assert module.config.mode == "static", "We are using dynamic quantization!"
        module.config.need_sample_for_static_init = 0

        scale_name = f"{name}.scale"
        zero_point_name = f"{name}.zero_point"

        if scale_name in data.keys():
            module.scale = data[scale_name].cuda()
            # f.write(scale_name + "\n")
        # else:
        #     f.write(scale_name + "\n")

        if zero_point_name in data.keys():
            module.zero_point = data[zero_point_name].cuda()
            # f.write(zero_point_name + "\n")
        else:
            module.zero_point = None
            # f.write(zero_point_name + "\n")
import torch
from tqdm import tqdm


from train.train_parameter import FakeQuantizer
from train.train_model import RotationEmbedding, RotationQuantLinear
from utils.utils import log 
from utils.convert_model import convert_model 


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


def trainable_static_rtn_fwrd(model, batch, model_args):
    """
    遍历 model，找到所有 FakeQuantizer 对象
    
    Args:
        model (nn.Module): 待遍历模型
    """
    model.eval() 

    layers = model.model.layers
    path = model_args.output_rotation_path
    data = torch.load(path, map_location="cpu")

    subset = find_qlayers(model, layers=[FakeQuantizer])
    for name in tqdm(subset, desc="(Trainable Static RtN Quant.) FakeQuantizer"):
        module = subset[name]
        assert module.config.mode == "static", "We are using dynamic quantization!"
        module.config.need_sample_for_static_init = 0

        scale_name = f"{name}.scale"
        zero_point_name = f"{name}.zero_point"
        num_bits_name = f"{name}.config.num_bits"

        if scale_name in data.keys():
            module.scale = data[scale_name].cuda()

        if zero_point_name in data.keys():
            module.zero_point = data[zero_point_name].round().cuda()
        else:
            module.zero_point = None

        # if num_bits_name in data.keys():
        #     module.config.num_bits = data[num_bits_name]
            
        if "outActQuant" in name or "qQuant" in name or "kQuant" in name or "vQuant" in name:
            module.config.need_sample_for_static_init = 16

    model.eval()
    with torch.no_grad(): 
        for i in tqdm(range(batch.size(0)), desc="Init scale and zero_point for static quant"):
            sample = batch[i].unsqueeze(0)  # 保持 batch 维度
            model(sample)
        log.info("✅ Init scale and zero_point ok!")

    # if model_args.convert_model_path is not None:
    #     convert_model(model, model_args.convert_model_path)

        
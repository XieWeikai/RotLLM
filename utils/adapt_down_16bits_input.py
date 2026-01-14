import torch
from tqdm import tqdm
import numpy as np


from train.train_parameter import FakeQuantizer
from train.train_model import RotationQuantLinear
from utils.utils import log
from train.config import ActivationQuantizeConfig


@torch.no_grad()
def adapt_choose_down_16bits_input(model, batch, batch_find_threshold, ptq_args, local_rank=None):
    if local_rank is None:
        local_rank = 0
    activations = {}
    hooks = []

    def get_activation_hook(name):
        def hook_fn(module, input, output):
            activations[name] = input[0].detach().cpu()
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, RotationQuantLinear) and "down_proj" in name:
            hook = module.register_forward_hook(get_activation_hook(name))
            hooks.append(hook)
    
    with torch.no_grad():
        _ = model(batch_find_threshold)

    """移除所有钩子"""
    for hook in hooks:
        hook.remove()
    hooks.clear()


    total_down = 0
    modified_down = 0

    activation_threshold_dict = []


    # 选择是否需要提高 down 层输入激活量化位宽为 16 bit
    layers = model.model.layers
    for i in tqdm(range(len(layers)), desc="Adaptive selection down_proj 16bit input activation", disable=not (local_rank == 0)):
        layer = layers[i]
        for name, module in layer.named_modules():
            all_name = f"model.layers.{i}." + name
            if not (all_name in activations):
                continue
            
            total_down += 1
            
            original_activation = activations[all_name].to("cuda") 
            quantized = module.actQuant(original_activation)

            original_activation = original_activation.cpu()
            quantized = quantized.cpu()

            # 计算误差
            error = (quantized - original_activation).abs()
            eps = 1e-8
            ratio = (error / (original_activation.abs() + eps)).sum() / original_activation.numel()
            ratio = ratio.cpu()          
            ratio = ratio.item()  

            activation_threshold_dict.append((ratio, all_name, module.actQuant.config))


    ratio_list = [x[0] for x in activation_threshold_dict]
    ratio_list_sorted = sorted(ratio_list, reverse=True)
    idx = int(np.ceil(ptq_args.adapt_R4_percentage * len(ratio_list_sorted))) - 1  # ceil 保证向上取整
    
    if idx == -1:
        threshold_ratio = 1.5
    else:
        threshold_ratio = ratio_list_sorted[idx]
    if local_rank == 0:
        log.info(f"idx: {idx}")
        log.info(f"threshold_ratio: {threshold_ratio}")
        print(ratio_list_sorted)

    for i in range(len(activation_threshold_dict)):
        ratio_i = activation_threshold_dict[i][0]
        all_name_i = activation_threshold_dict[i][1]
        config_i = activation_threshold_dict[i][2]

        if ratio_i >= threshold_ratio:
            config_i.num_bits = 16
            modified_down += 1     
            if local_rank == 0:
                log.info(f"{all_name_i}: Improve quantization precision to 16bit!")

    for name, module in model.named_modules():
        if isinstance(module, FakeQuantizer):
            module.config.need_sample_for_static_init = 16
            if hasattr(module, "scale"):
                delattr(module, "scale")
            if hasattr(module, "zero_point"):
                delattr(module, "zero_point")
            if hasattr(module, "xmax"):
                delattr(module, "xmax")
            if hasattr(module, "xmin"):
                delattr(module, "xmin")
            if isinstance(module.config, ActivationQuantizeConfig):
                module.config.init_type = "mean"

    
    layers = model.model.layers
    for i in range(len(layers)):
        layers[i].mlp.down_proj.actQuant.config.init_type = "maxmin"


    if local_rank == 0:
        log.info("\n===== Adaptive selection down_proj 16bit input activation Summary =====")
        log.info(f"16bit input activation: {modified_down}/{total_down} "
            f"({modified_down / max(total_down, 1) * 100:.2f}%) added")
        log.info("Improve ok!")

    
    # 后续如果需要做混合精度量化，这里需要进行量化配置的修改
    if ptq_args.adaptive_mixed_precision:
        for name, module in model.named_modules():
            if isinstance(module, FakeQuantizer):
                if isinstance(module.config, ActivationQuantizeConfig):
                    module.config.init_type = "maxmin"

    if local_rank == 0:
        from utils.adapt_mix_precision import collect_fakequant_configs
        collect_fakequant_configs(model, "./txt/four.txt", True)


    with torch.no_grad(): 
        for i in tqdm(range(batch.size(0)), desc="Re-init scale and zero_point for static quant(Adaptive 16bit)", disable=not (local_rank == 0)):
            sample = batch[i].unsqueeze(0)  # 保持 batch 维度
            model(sample)
        if local_rank == 0:
            log.info(f"✅ Re-init scale and zero_point ok!")

    return model
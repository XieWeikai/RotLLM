import torch
import dataclasses
import os
from tqdm import tqdm
import numpy as np


from train.train_parameter import FakeQuantizer
from train.train_model import RotationQuantLinear
from utils.utils import log
from train.config import ActivationQuantizeConfig


def collect_fakequant_configs(model, filename=None, write_to_file=False, local_rank=None):
    """
    Find all FakeQuantizers.config in the model.
    If write_to_file is True, save them to a txt file.
    """
    if local_rank is None:
        local_rank = 0
    fq_dict = {}

    # 仅当需要写文件时创建目录并打开文件
    if write_to_file:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        f = open(filename, "w")

    for name, module in model.named_modules():
        if isinstance(module, FakeQuantizer):
            fq_dict[name] = module.config
            config_dict = dataclasses.asdict(module.config)
            if write_to_file:
                f.write(f"{name}: {config_dict}\n")

    if write_to_file:
        f.close()

    if write_to_file and local_rank == 0:
        log.info(f"Fakequant configs saved ok to {filename}!")
    return fq_dict



@torch.no_grad()
def adapt_modify_quantization_precision(model, adaptive_R4, batch, batch_find_threshold, ptq_args, local_rank=None):
    if local_rank is None:
        local_rank = 0
    activations = {}
    hooks = []

    def get_activation_hook(name):
        def hook_fn(module, input, output):
            activations[name] = input[0].detach().cpu()
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, RotationQuantLinear):
            hook = module.register_forward_hook(get_activation_hook(name))
            hooks.append(hook)
    
    with torch.no_grad():
        _ = model(batch_find_threshold)

    """移除所有钩子"""
    for hook in hooks:
        hook.remove()
    hooks.clear()

    total_activation_quantizer_list = []
    modified_activation_quantizer_counts = 0

    activation_threshold_dict = []

    layers = model.model.layers
    for i in tqdm(range(len(layers)), desc="Modify quant num_bits", disable=not (local_rank == 0)):
        layer = layers[i]
        for name, module in layer.named_modules():
            all_name = f"model.layers.{i}." + name
            if not (all_name in activations):
                continue
            
            prefix = all_name.rsplit('.', 1)[0]
            suffix = all_name.rsplit('.', 1)[-1]

            if suffix in ["q_proj", "k_proj", "v_proj"]:
                if all(prefix + i not in total_activation_quantizer_list for i in [".q_proj", ".k_proj", ".v_proj"]):
                    total_activation_quantizer_list.append(all_name)
                else:
                    continue
            elif suffix in ["up_proj", "gate_proj"]:
                if all(prefix + i not in total_activation_quantizer_list for i in [".up_proj", ".gate_proj"]):
                    total_activation_quantizer_list.append(all_name)
                else:
                    continue
            else:
                total_activation_quantizer_list.append(all_name)

            
            original_activation = activations[all_name].to("cuda") 
            quantized = module.actQuant(original_activation)

            original_activation = original_activation.cpu()
            quantized = quantized.cpu()

            # 计算误差
            error = (quantized - original_activation).abs()
            eps = 1e-8
            ratio = (error / (original_activation.abs() + eps)).sum() / original_activation.numel()
            # original_activation = original_activation.cpu()
            ratio = ratio.cpu()          # tensor 在 CPU 上
            ratio = ratio.item()   # 转为 Python float

            activation_threshold_dict.append((ratio, all_name, module.actQuant.config))


    ratio_list = [x[0] for x in activation_threshold_dict]
    ratio_list_sorted = sorted(ratio_list, reverse=True)
    idx = int(np.ceil(ptq_args.adapt_activation_percentage * len(ratio_list_sorted))) - 1  # ceil 保证向上取整

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

        if ratio_i >= threshold_ratio and config_i.num_bits < 8:
            config_i.num_bits = 8
            modified_activation_quantizer_counts += 1
            if local_rank == 0:
                log.info(f"{all_name_i}: Modify activation!")
            

    
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


    # 如果前面做了 R4 的自适应选择，那么部分无 R4 旋转的 down_proj 层的 activation 量化参数只能使用 maxmin 初始化方式
    no_R4_count = 0            
    if ptq_args.adaptive_online_rotation_R4:
        layers = model.model.layers
        for i in range(len(layers)):
            if not adaptive_R4[i]:
                layers[i].mlp.down_proj.actQuant.config.init_type = "maxmin"
                no_R4_count += 1

    if ptq_args.adaptive_down_input_activation_16bits:
        layers = model.model.layers
        for i in range(len(layers)):
            layers[i].mlp.down_proj.actQuant.config.init_type = "maxmin"
            no_R4_count += 1


    if local_rank == 0:
        log.info("\n===== Quantization Modification Summary =====")
        log.info(f"Activation quantizers(Adaptive R4 Stage): {no_R4_count}/{len(total_activation_quantizer_list)} "
            f"({no_R4_count / max(len(total_activation_quantizer_list), 1) * 100:.2f}%) modified")
        log.info(f"Activation quantizers(Adaptive precision Stage): {modified_activation_quantizer_counts}/{len(total_activation_quantizer_list)} "
            f"({modified_activation_quantizer_counts / max(len(total_activation_quantizer_list), 1) * 100:.2f}%) modified")
        log.info(f"Activation quantizers(Summary): {no_R4_count + modified_activation_quantizer_counts}/{len(total_activation_quantizer_list)} "
            f"({no_R4_count + modified_activation_quantizer_counts / max(len(total_activation_quantizer_list), 1) * 100:.2f}%) modified")
        log.info("Modify ok!")

    if local_rank == 0:
        collect_fakequant_configs(model, "./txt/three.txt", True)


    with torch.no_grad(): 
        for i in tqdm(range(batch.size(0)), desc="Re-init scale and zero_point for static quant(Adaptive precision)", disable=not (local_rank == 0)):
            sample = batch[i].unsqueeze(0)  # 保持 batch 维度
            model(sample)
        if local_rank == 0:
            log.info(f"✅ Re-init scale and zero_point ok!")
    return model
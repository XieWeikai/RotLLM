import torch
import dataclasses
import os
from tqdm import tqdm
import numpy as np


from train.train_parameter import FakeQuantizer
from train.train_model import RotationQuantLinear
from utils.utils import log


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




# def rotate_weight(w, b, rotation_pos, R_pre, R_post):
#     if rotation_pos in ["pre", "around"]:
#         assert w.shape[1] % R_pre.weight.shape[0] == 0, "Input dim should be multiple of R_pre dim"
#         num_blocks = w.shape[1] // R_pre.weight.shape[0]

#         w_dtype = w.dtype
#         w_device = w.device
#         w = w.view(w.shape[0], num_blocks, R_pre.weight.shape[0])
#         # w = self.R_pre(w)
#         w = (w.to(R_pre.weight.dtype) @ R_pre.weight.to(device=w_device)).to(dtype=w_dtype)
#         # w = (w.to(dtype=torch.float64) @ self.R_pre.weight.to(dtype=torch.float64, device=w_device)).to(dtype=w_dtype) 
#         w = w.view(w.shape[0], num_blocks * R_pre.weight.shape[0])

#     if rotation_pos in ["post", "around"]:
#         assert w.shape[0] % R_post.weight.shape[0] == 0, "Output dim(weight) should be multiple of R_post dim"
#         num_blocks = w.shape[0] // R_post.weight.shape[0]

#         w_dtype = w.dtype
#         w_device = w.device
#         w = w.T
#         w = w.view(w.shape[0], num_blocks, R_post.weight.shape[0])
#         # w = self.R_post(w)
#         w = (w.to(R_post.weight.dtype) @ R_post.weight.to(device=w_device)).to(dtype=w_dtype)
#         # w = (w.to(dtype=torch.float64) @ self.R_post.weight.to(dtype=torch.float64, device=w_device)).to(dtype=w_dtype)
#         w = w.view(w.shape[0], num_blocks * R_post.weight.shape[0])
#         w = w.T
#         if b is not None:
#             assert b.shape[0] % R_post.weight.shape[0] == 0, "Output dim(bias) should be multiple of R_post dim"
#             b_dtype = b.dtype
#             b_device = b.device
#             # b = self.R_post(b.view(num_blocks, -1))
#             b = (b.to(R_post.weight.dtype).view(num_blocks, -1) @ R_post.weight.to(device=b_device)).to(dtype=b_dtype)
#             # b = (b.to(dtype=torch.float64).view(num_blocks, -1) @ self.R_post.weight.to(dtype=torch.float64, device=b_device)).to(dtype=b_dtype)
#             b = b.view(-1)
#     return w


@torch.no_grad()
def adapt_modify_fakequant_configs(model, batch_find_threshold, ptq_args, local_rank=None):
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
    
    # 前向传播触发 hooks
    with torch.no_grad():
        _ = model(batch_find_threshold)

    total_activation_quantizer_list = []
    modified_activation_quantizer_counts = 0
    # total_weight_quantizer_counts = 0
    # modified_weight_quantizer_counts = 0

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
            

            # b = module.linear.bias.data if module.linear.bias is not None else None
            # w = module.linear.weight.data
            # original_weight = rotate_weight(w, b, module.rotation_pos, module.R_pre, module.R_post)
            # quantized = module.weightQuant(original_weight)
            
            # # original_max_val = original_weight.to(torch.float32).max()
            # # original_min_val = original_weight.to(torch.float32).min()
            # # quantized_max_val = quantized.to(torch.float32).max()
            # # quantized_min_val = quantized.to(torch.float32).min()
            # # ratio1 = (original_max_val - quantized_max_val).abs() / original_max_val.abs()
            # # ratio2 = (original_min_val - quantized_min_val).abs() / original_min_val.abs()
            # # ratio = max(ratio1, ratio2)

            # x = original_weight.to(torch.float32).abs().flatten()
            # k = max(1, int(x.numel() * 0.001))  # top 0.1%
            # topk_values, _ = torch.topk(x, k)
            # threshold = topk_values[-1]  # 最小的 top 值
            # max_val = x.max()
            # ratio = threshold / max_val

            # total_weight_quantizer_counts += 1
            # if ratio > 1.0 and module.weightQuant.config.num_bits < 8:
            #     module.weightQuant.config.num_bits = 8
            #     if local_rank == 0:
            #         print(f"{all_name}: Modify weight!")
            #     modified_weight_quantizer_counts += 1
            
        
    """移除所有钩子"""
    for hook in hooks:
        hook.remove()
    hooks.clear()
    
    for name, module in model.named_modules():
        if isinstance(module, FakeQuantizer):
            module.config.need_sample_for_static_init = 16
            if hasattr(module, "scale"):
                delattr(module, "scale")
            if hasattr(module, "zero_point"):
                delattr(module, "zero_point")
            module.config.init_type = "mean"

    if local_rank == 0:
        log.info("\n===== Quantization Modification Summary =====")
        log.info(f"Activation quantizers: {modified_activation_quantizer_counts}/{len(total_activation_quantizer_list)} "
            f"({modified_activation_quantizer_counts / max(len(total_activation_quantizer_list), 1) * 100:.2f}%) modified")
        # log.info(f"Weight quantizers:     {modified_weight_quantizer_counts}/{total_weight_quantizer_counts} "
        #     f"({modified_weight_quantizer_counts / max(total_weight_quantizer_counts, 1) * 100:.2f}%) modified")
        log.info("Modify ok!")
    return model
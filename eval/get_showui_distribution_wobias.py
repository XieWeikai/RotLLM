# get photo info from json
from collections import defaultdict
from functools import partial
import gc
import json

from tqdm import tqdm
from PIL import Image

import torch
import numpy as np

from args import args
args.no_quantize=True

# from evaluate import load
from datasets import load_dataset

from copy import deepcopy


from utils.model import LLMNPUShowUIModel

model = LLMNPUShowUIModel(args.tokenizer_name, args.model_name, args=args)

act_dict = defaultdict(dict)

def flatten_act_dict(act_dict):
    for layer, scales in act_dict.items():
        if isinstance(scales, list):
            try:
                all_acts = np.array(scales).reshape(-1)
            except ValueError:
                all_acts = [np.array(scale).reshape(-1) for scale in scales]
            all_acts = np.concatenate(all_acts)
            act_dict[layer] = all_acts
        else:
            act_dict[layer] = flatten_act_dict(scales)
            print(layer)
        gc.collect()

    return act_dict

def get_act_percentage(act_dict: dict, threshold: float):
    assert 0 <= threshold <= 1
    percentage = 1 - threshold
    act_percentage = {}
    for layer, scales in act_dict.items():
        if not isinstance(scales, dict):
            all_acts_flattened = scales
            percentage_index = int(len(all_acts_flattened) * percentage) - 1
            nth_percentile_value = np.partition(all_acts_flattened, percentage_index)[
                percentage_index
            ]
            act_percentage[layer] = float(nth_percentile_value)
        else:
            print(layer)
            act_percentage[layer] = get_act_percentage(scales, threshold)
    return act_percentage


@torch.no_grad()
def get_static_decoder_layer_scales_distribution(
    model,
    num_samples=32,
):

    def stat_io_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if name not in act_dict or "input" not in act_dict[name]:
            act_dict[name]["input"] = []
        act_dict[name]["input"].append(x.clone().detach().cpu().numpy())
        if isinstance(y, tuple):
            y = y[0]

        ty = y.clone().detach().cpu()
        # 去除 bias（只针对 nn.Linear）
        if isinstance(m, torch.nn.Linear) and m.bias is not None:
            # print(name + str(".wobias"))
            # print(y.shape)
            
            bias = m.bias.clone().detach().view(1, -1)  # shape [1, out_features]
            ty = ty - bias.to(ty.device)

        if "output" not in act_dict[name]:
            act_dict[name]["output"] = []
        act_dict[name]["output"].append(ty.detach().cpu().numpy())

    hooks = []
    for name, m in model.model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(partial(stat_io_hook, name=name)))
        
    print("Collecting activation scales...")

    import ast
    from tqdm import tqdm
    from datasets import load_dataset

    # dataset = load_dataset("/data/share/datasets_roci/ScreenSpot/", split="test", cache_dir="/data/xudaliang/huggingface_cache/")  # noqa E501
    dataset = load_dataset("/data/share/datasets_roci/ScreenSpot/", split="test")  # noqa E501

    # 打乱数据集，设置随机种子以确保可重复性  
    shuffled_dataset = dataset.shuffle(seed=42)  

    # 随机选择前 100 个样本  
    random_sampled_dataset = shuffled_dataset.select(range(num_samples))  


    mobile_data_length = 0
    correct = 0

    with tqdm(total=len(random_sampled_dataset)) as pbar:
        
        pbar.set_description("ScreenSpotPipelineImpl Processing:")
        for data in random_sampled_dataset:
            pc_or_mobile = data["file_name"].split("_")[0]
            if data["data_type"] in ["text"] and pc_or_mobile == "mobile":
                mobile_data_length += 1
                point = []
                out_text = ""
                # try:
                out_text = model.infer(data["image"], data["instruction"], None)[0]
                point = ast.literal_eval(out_text)
                # except:
                #     print(data["file_name"], "ast parse failed", out_text)
                #     pbar.update(1)
                #     continue
                bbox = data["bbox"]
                x_min, y_min, x_max, y_max = bbox
                px, py = point

                print(px, py)

                is_inside = (x_min <= px <= x_max) and (y_min <= py <= y_max)
                if is_inside:
                    correct += 1
                else:
                    print(data["file_name"], "position failed", bbox, point)
            pbar.update(1)
        
        print("acc: ", correct / mobile_data_length)
        

    for hook in hooks:
        hook.remove()

    return act_dict



def get_act_distribution_stat(act_dict):
    act_distribution = {}
    for layer, scales in act_dict.items():
        if not isinstance(scales, dict):
            act_distribution[layer] = {
                "mean": float(np.mean(scales)),
                "std": float(np.std(scales)),
            }
        else:
            act_distribution[layer] = get_act_distribution_stat(scales)
    return act_distribution


get_static_decoder_layer_scales_distribution(model, 64)

print("begin_flatten")
act_dict = flatten_act_dict(act_dict)
print("finish flatten")

# origin model scale
print("begin_calculate")
print("get act 0")
ori_scale = get_act_percentage(act_dict, 0)
# scale after remove top 0.1% outliers
print("get act 0.001")
top_0_1_scale = get_act_percentage(act_dict, 0.001)
# get mean and std of all scales
print("get act distribution")
all_stat = get_act_distribution_stat(act_dict)
res_dict = {"ori": ori_scale, "top_0_1": top_0_1_scale, "all_stat": all_stat}
with open(args.output_file, "w") as f:
    json.dump(res_dict, f, indent=4, ensure_ascii=False)


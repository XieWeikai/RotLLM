import torch
from tqdm import tqdm
import os
from train.train_model import RotationQuantLinear, RotationEmbedding
import torch.nn as nn

def is_target_layer(name):
    """
    判断是否是需要量化的 Linear 层。
    排除 lm_head。
    目标：layers.X 中的 q, k, v, o, gate, up, down
    """ 
    
    # Qwen 的结构通常是 model.layers.X.self_attn.q_proj 等
    target_keywords = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    if any(k in name for k in target_keywords):
        return True
    return False


def convert_executorch_model(model, convert_model_path):
    new_state_dict = {}
    
    print("开始转换权重...")
    open("./txt/convert_log.txt", "w").close()
    
    # 遍历所有参数
    for name, module in tqdm(model.named_modules()):
        if not (isinstance(module, (RotationQuantLinear, nn.Embedding)) or "norm" in name):
            continue

        with open("./txt/convert_log.txt", "a") as f:
            f.write(f"{name}\n")
        
        # 如果是需要量化的层
        if is_target_layer(name):
            assert isinstance(module, RotationQuantLinear), "False"

            # 1. 拿到Weight Scale
            # param shape: [out_dim, in_dim] (Linear层的默认存储)
            weight = module.linear.weight
            bias = module.linear.bias
            scale = module.weightQuant.scale # 获得训练得到的该weight的scale
                 
            # 3. 构建保存的 Key 名称
            base_name = name
            if base_name not in new_state_dict:
                new_state_dict[base_name] = {}
            
            # 写入权重 (FP32)
            new_state_dict[base_name]['weight'] = weight.contiguous().cpu()

            # 写入 bias (FP32)
            new_state_dict[base_name]['bias'] = bias.contiguous().cpu() if bias is not None else None

            # 写入 Scale (FP32)
            new_state_dict[base_name]['scale'] = scale.contiguous().cpu()


            if f"{base_name}.input" not in new_state_dict:
                new_state_dict[f"{base_name}.input"] = {}
            # 写入 Input Scale (FP32)
            input_scale = module.actQuant.scale # input scale
            new_state_dict[f"{base_name}.input"]['scale'] = input_scale.contiguous().cpu()

            # 写入 Input Zero_point (FP32)
            input_zero_point = module.actQuant.zero_point # input zero_point
            new_state_dict[f"{base_name}.input"]['zero_point'] = input_zero_point.contiguous().cpu()
        elif "q_norm" not in name and "k_norm" not in name and "norm" in name:
            base_name = name
            if base_name not in new_state_dict:
                new_state_dict[base_name] = {}
        
            # 写入权重 (int32 --> uint16)
            # new_state_dict[base_name]['weight'] = torch.full_like(module.weight, 65535, dtype=torch.int32).contiguous().cpu()
            new_state_dict[base_name]['weight'] = module.weight.contiguous().cpu()

            # # 写入 scale (FP32)
            # new_state_dict[base_name]['scale'] = torch.tensor(3.05176e-5, dtype=torch.float32).contiguous().cpu()

            # # 写入 zero_point (int32)
            # new_state_dict[base_name]['zero_point'] = torch.tensor(32768, dtype=torch.int32).contiguous().cpu()
        elif "head" in name:
            base_name = name
            if base_name not in new_state_dict:
                new_state_dict[base_name] = {}
        
            new_state_dict[base_name]['weight'] = module.linear.weight.contiguous().cpu()
        elif "embed" in name:
            base_name = name
            if base_name not in new_state_dict:
                new_state_dict[base_name] = {}
            new_state_dict[base_name]['weight'] = module.weight.contiguous().cpu()


    dir_path = os.path.dirname(convert_model_path)  
    os.makedirs(dir_path, exist_ok=True)  
    torch.save(new_state_dict, convert_model_path)
    print("权重转换完成...")



def check_pth(path):
    # 加载 .pth 文件（state_dict）
    state_dict = torch.load(path, map_location="cpu")

    # 输出所有 key, value.shape
    for k, v in state_dict.items():
        # print(k, v.shape)
        print(k)


def check_qkv_input_scale(pth_path):
    # 加载 state_dict
    state_dict = torch.load(pth_path, map_location="cpu")

    # 找到所有 layer 的名字
    layers = set()
    for key in state_dict.keys():
        # 假设 key 是类似 "model.layers.0.self_attn.q_proj.input.scale"
        if "self_attn" in key and "input" in key:
            layer_name = ".".join(key.split(".")[:4])  # "model.layers.0.self_attn"
            layers.add(layer_name)

    layers = sorted(layers)
    for layer in layers:
        q_scale = state_dict.get(f"{layer}.q_proj.input")["scale"]
        k_scale = state_dict.get(f"{layer}.k_proj.input")["scale"]
        v_scale = state_dict.get(f"{layer}.v_proj.input")["scale"]

        if q_scale is None or k_scale is None or v_scale is None:
            print(f"{layer}: missing some scales")
            continue

        # 检查是否相等
        if torch.allclose(q_scale, k_scale) and torch.allclose(q_scale, v_scale):
            print(f"{layer}: q/k/v input scales are equal")
        else:
            print(f"{layer}: q/k/v input scales are NOT equal")
            print(f"  q: {q_scale}")
            print(f"  k: {k_scale}")
            print(f"  v: {v_scale}")


if __name__ == "__main__":
    check_pth("/data/zjh/model_pth/Qwen3-1.7B-rotated.pth")
    # from transformers import AutoModel
    # model = AutoModel.from_pretrained("/data/share/Qwen2.5-3B")
    # print(model)
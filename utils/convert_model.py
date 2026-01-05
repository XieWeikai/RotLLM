import torch
from tqdm import tqdm
import os


def quantize_given_scale(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
     使用给定的 scale 对权重张量进行量化（per-tensor）。
    Args:
        w (torch.Tensor): 待量化的权重张量，dtype 可为 float32。
        scale (torch.Tensor): 已知的缩放因子，shape=[1]，dtype float32。
    
    Returns:
        torch.Tensor: 量化后的权重张量，int8
    """
    if scale.numel() != 1:
        raise ValueError(f"scale should be a tensor of shape [1], got {scale.shape}")
    
    # 可以用cuda加速，不需要的话去掉这行
    w = w.to("cuda")
    
    # 量化
    w = w / scale.to(w.device)  # 除以给定 scale
    w = w.round_()  # 四舍五入
    
    # 转换为int8
    w_q = w.to("cpu").type(torch.int8)
    
    return w_q



def is_target_layer(name):
    """
    判断是否是需要量化的 Linear 层。
    排除 lm_head。
    目标：layers.X 中的 q, k, v, o, gate, up, down
    """
    if "lm_head" in name:
        return False
    
    # Qwen 的结构通常是 model.layers.X.self_attn.q_proj 等
    target_keywords = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    if any(k in name for k in target_keywords) and "weight" in name:
        return True
    return False


def get_module_by_name(model, name):
    parts = name.split('.')
    module = model
    for p in parts[:-2]:  # 最后一个是 weight/bias，不是 module
        if p.isdigit():          # 处理 layers.0 这种数字 index
            module = module[int(p)]
        else:
            module = getattr(module, p)
    return module


def convert_model(model, convert_model_path):
    state_dict = model.state_dict()
    new_state_dict = {}
    
    print("开始转换权重...")
    open("./txt/convert_log.txt", "w").close()
    
    # 遍历所有参数
    for name, param in tqdm(state_dict.items()):
        with open("./txt/convert_log.txt", "a") as f:
            f.write(f"{name}\n")

        # 如果是需要量化的层
        if is_target_layer(name):
            linear_module = get_module_by_name(model, name)

            # 1. 拿到Weight Scale
            # param shape: [out_dim, in_dim] (Linear层的默认存储)
            scale_val = linear_module.weightQuant.scale # 获得训练得到的该weight的scale
            
            # 2. 量化权重
            # 此时 w_quant shape: [out_dim, in_dim]
            w_quant = quantize_given_scale(param, scale_val)
            
            
            # 3. 构建保存的 Key 名称
            base_name = name.replace(".linear.weight", "")
            
            # 写入量化后的权重 (Int8)
            new_state_dict[f"{base_name}.weight"] = w_quant.cpu()
            
            # 写入 Scale (FP32)
            new_state_dict[f"{base_name}.scale"] = scale_val.cpu()
            
            # 写入 Input Scale (FP32)
            input_scale = linear_module.actQuant.scale # input scale
            new_state_dict[f"{base_name}.input_scale"] = input_scale
            
            # 写入 Output Scale (FP32) 
            output_scale = linear_module.outActQuant.scale # output scale
            new_state_dict[f"{base_name}.output_scale"] = output_scale
            
        else:
            # 不需要量化的层（如 Norm, Embeddings, lm_head），直接复制
            # 保持原样 (FP16/FP32)
            if "weight" not in name:
                continue
            if "linear" in name:
                name = name.replace(".linear", "")
            new_state_dict[name] = param.to(torch.float32).cpu()

    dir_path = os.path.dirname(convert_model_path)  
    os.makedirs(dir_path, exist_ok=True)  
    torch.save(new_state_dict, convert_model_path)
    print("权重转换完成...")



def check_pth():
    # 加载 .pth 文件（state_dict）
    state_dict = torch.load("/data/zjh/model_pth/Qwen3-rotated.pth", map_location="cpu")

    # 输出所有 key, value.shape
    for k, v in state_dict.items():
        print(k, v.shape)

if __name__ == "__main__":
    check_pth()
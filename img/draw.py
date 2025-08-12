import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import copy

from fakequant.linear import replace_linear_with_fakequant, FakeQuantLinear
import rotate

# ========= 配置 =========
# MODEL_PATH_ORIG = "/root/autodl-tmp/RotLLM/LLM/phi-3-mini-4k-instruct"
MODEL_PATH_ORIG = "/root/autodl-tmp/RotLLM/LLM/Qwen2.5-1.5B-Instruct"

device = "cuda:0"
dtype = torch.float32

# ========= 激活 Hook 函数 =========
activation_dict = {}  # key: "layer0.q_proj"  value: list of np.array激活值

def get_activation_hook(name):
    def hook(module, input, output):
        act = output.detach().cpu().flatten().numpy()
        if name not in activation_dict:
            activation_dict[name] = []
        activation_dict[name].append(act)
    return hook

# ========= 加载 tokenizer 和数据 =========
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_ORIG)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
dataset = dataset.filter(lambda example: len(example['text'].strip()) > 0)

# ========= 加载模型并旋转 + 量化 =========
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH_ORIG, device_map=device, torch_dtype=dtype)
model.eval()

num_layers = model.config.num_hidden_layers
dim = model.config.hidden_size
num_heads = model.config.num_attention_heads
head_dim = dim // num_heads

# 生成 Hadamard 旋转矩阵并旋转
R = rotate.get_orthogonal_matrix(dim, mode="hadamard", device=device)
R_v = [rotate.get_orthogonal_matrix(head_dim, mode="hadamard", device=device) for _ in range(num_layers)]
rotate.rotate_model(model, R, R_v)

# 应用 int8 假量化
# replace_linear_with_fakequant(model, 8)

# ========= 注册 Hook（前两层 Linear）=========
hooks = []
for i in range(1):
    block = model.model.layers[i]
    for name, module in block.named_modules():
        if isinstance(module, (torch.nn.Linear, FakeQuantLinear)):
            full_name = f"layer{i}.{name}"  # 完整唯一标识
            print(f"  -> Registering hook on: {full_name}")
            hooks.append(module.register_forward_hook(get_activation_hook(full_name)))

# ========= 前向传播（仅触发一次）=========
sample_text = dataset[0]['text']
encodings = tokenizer(sample_text, return_tensors='pt', truncation=True, max_length=512)
input_ids = encodings.input_ids.to(device)
with torch.no_grad():
    _ = model(input_ids)

# pre_activation_dict = copy.deepcopy(activation_dict)
# activation_dict = {}

# rotate.rotate_model(model, R, R_v)
# with torch.no_grad():
#     _ = model(input_ids)

# for name in pre_activation_dict:
#     pre_act = np.concatenate(pre_activation_dict[name])
#     post_act = np.concatenate(activation_dict[name])
    
#     if pre_act.shape != post_act.shape:
#         print(f"[{name}] shape mismatch!")
#         continue
    
#     diff = np.abs(pre_act - post_act)
#     # diff = post_act
#     max_diff = diff.max()
#     mean_diff = diff.mean()
    
#     print(f"[{name}] max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}")



# ========= 绘制激活值分布图 =========
module_names = sorted(activation_dict.keys(), key=lambda x: (
    int(x.split('.')[0][5:]),  # layer index，确保按层排序
    x.split('.')[1]            # 子模块名字典序或你自定义顺序
))

name_to_idx = {name: idx for idx, name in enumerate(module_names)}

x_vals = []
y_vals = []
z_vals = []

for name in module_names:
    acts = np.concatenate(activation_dict[name])  # 合并当前模块所有激活
    for idx, val in enumerate(acts):
        x_vals.append(name_to_idx[name])
        y_vals.append(idx)
        z_vals.append(val)

x_vals = np.array(x_vals)
y_vals = np.array(y_vals)
z_vals = np.array(z_vals)

# 3D绘图
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap='coolwarm', s=1)

ax.set_title("Activation Distribution per Minimal Linear Layer")
ax.set_xlabel("Minimal Linear Layer (q_proj, k_proj, ...)")
ax.set_ylabel("Activation Index")
ax.set_zlabel("Activation Value")

ax.set_xticks(list(name_to_idx.values()))
ax.set_xticklabels(module_names, rotation=45, ha='right')

plt.tight_layout()
plt.savefig("activation_distribution.png")
print("图像已保存为 activation_distribution.png")

print("Show success!")

# ========= 清理 hook =========
for h in hooks:
    h.remove()

# ========= 清理资源 =========
model.cpu()
del model
torch.cuda.empty_cache()



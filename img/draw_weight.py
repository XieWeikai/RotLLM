import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

from fakequant.linear import replace_linear_with_fakequant, FakeQuantLinear
import rotate

# ========= 配置 =========
MODEL_PATH_ORIG = "/root/autodl-tmp/RotLLM/LLM/phi-3-mini-4k-instruct"
device = "cuda:0"
dtype = torch.float32

# ========= 加载模型和 tokenizer =========
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_ORIG)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH_ORIG, device_map=device, torch_dtype=dtype)
model.eval()

num_layers = model.config.num_hidden_layers
dim = model.config.hidden_size
num_heads = model.config.num_attention_heads
head_dim = dim // num_heads

# ========= 获取并收集每层的权重 =========
weight_dict = {}

for i in range(1):  # 只看前1层
    block = model.model.layers[i]
    for name, module in block.named_modules():
        if isinstance(module, (torch.nn.Linear, FakeQuantLinear)):
            full_name = f"layer{i}.{name}"
            weight = module.weight.detach().cpu().flatten().numpy()
            weight_dict[full_name] = weight
            # print(f"Collected weights from: {full_name}")

# ========= 绘图数据准备 =========
module_names = sorted(weight_dict.keys(), key=lambda x: (
    int(x.split('.')[0][5:]),  # layer index
    x.split('.')[1]
))
name_to_idx = {name: idx for idx, name in enumerate(module_names)}

x_vals = []
y_vals = []
z_vals = []

for name in module_names:
    weights = weight_dict[name]
    for idx, val in enumerate(weights):
        x_vals.append(name_to_idx[name])
        y_vals.append(idx)
        z_vals.append(val)

x_vals = np.array(x_vals)
y_vals = np.array(y_vals)
z_vals = np.array(z_vals)

# ========= 3D 可视化权重分布 =========
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap='coolwarm', s=1)

ax.set_title("Weight Distribution per Minimal Linear Layer")
ax.set_xlabel("Minimal Linear Layer (q_proj, k_proj, ...)")
ax.set_ylabel("Weight Index")
ax.set_zlabel("Weight Value")

ax.set_xticks(list(name_to_idx.values()))
ax.set_xticklabels(module_names, rotation=45, ha='right')

ax.set_xticks(list(name_to_idx.values()))
ax.set_xticklabels(module_names, rotation=45, ha='right')

plt.subplots_adjust(bottom=0.3, left=0.1, right=0.9, top=0.9)

# plt.tight_layout()
plt.savefig("weight_distribution.png")
print("图像已保存为 weight_distribution.png")

# ========= 清理资源 =========
model.cpu()
del model
torch.cuda.empty_cache()

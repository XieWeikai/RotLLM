# RotLLM
This is an implementation of [SpinQuant](https://arxiv.org/abs/2405.16406) and [QuaRot](https://arxiv.org/abs/2404.00456) for different models like Qwen. We are not intented to do exactly the same things as SpinQuant and QuaRot, instead we provide a framework to customize rotation operations for any models you want to use.

## Requirements
```python
# CUDA == 12.1
# Python == 3.9
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.57.1
pip install protobuf==6.32.0
pip install datasets==2.20.0
pip install accelerate==1.10.1
pip install tensorboard==2.20.0
```


## Quick Adaptation to New Models

The current codebase fully supports Llama, Qwen2.5, SmolLM2, and Qwen3.
Adapting a new model is straightforward: it only requires adding a few lines of code to implement the corresponding modeling.py. Taking Llama as an example, we first reuse the MLP implementation from [Hugging Face Transformers](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/llama/modeling_llama.py):

```python
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
```

We do not need to modify the overall logic of the MLP. Instead, we only insert a few lines to incorporate online rotation (R4), resulting in a customized MLP module:

```python
class LlamaMLPWithR4(nn.Module):
    def __init__(self, module: LlamaMLP, R4):
        super().__init__()
        self.config = module.config
        self.hidden_size = module.hidden_size
        self.intermediate_size = module.intermediate_size
        self.gate_proj = module.gate_proj
        self.up_proj = module.up_proj
        self.down_proj = module.down_proj
        self.act_fn = module.act_fn
        self.R4 = R4

    def forward(self, x):
        # We modify (add R4)
        gated_activation = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        gated_activation_dtype = gated_activation.dtype
        gated_activation_device = gated_activation.device
        down_proj = self.down_proj((gated_activation.to(dtype = self.R4.weight.dtype) @ self.R4.weight.to(gated_activation_device)).to(dtype = gated_activation_dtype))

        return down_proj
```


 We can then directly reuse the provided apply_R4_change_model function, with only minor modifications to the class names:
 
```python
def apply_R4_change_model(model, R4_list, local_rank=None):
    """
        Replace LlamaMLP with LlamaMLPWithR4
    """
    if local_rank is None:
        local_rank = 0
    layers = model.model.layers
    for i in tqdm(range(len(layers)), desc="Replace LlamaMLP with LlamaMLPWithR4", disable=not (local_rank == 0)):
        layer = layers[i]
        for name, module in layer.named_children():
            if isinstance(module, LlamaMLP):
                # Take out the R4 of the corresponding layer from the list.
                R4 = R4_list[i]
                setattr(layer, name, LlamaMLPWithR4(module, R4))
```
Finally, place the implemented file under the [modeling/](https://github.com/XieWeikai/RotLLM/tree/dev-zjh/modeling), named according to the target model architecture.



## Rotation Matrix Configuration
We provide flexible options for constructing rotation matrices in [prepare_model.py](https://github.com/XieWeikai/RotLLM/blob/dev-zjh/runner/prepare_model.py). Specifically, users can choose between "identity" (identity matrix), "random" (random orthogonal matrix), and "hadamard" (random Hadamard matrix) modes to control the type of rotation applied in the model.

```python
R1 = LearnRotateModule(get_orthogonal_matrix(dim, mode="hadamard", device=device))
R2 = [LearnRotateModule(get_orthogonal_matrix(head_dim, mode="hadamard", device=device)) for _ in range(num_layers)]
R3 = [NoLearnRotateModule(get_orthogonal_matrix(head_dim, mode="identity", device=device)) for _ in range(num_layers)]
R4 = [NoLearnRotateModule(get_orthogonal_matrix(hidden_dim, mode="identity", device=device)) for _ in range(num_layers)]  
```


## Usage
```python
git clone https://github.com/XieWeikai/RotLLM.git
cd RotLLM
git checkout dev-zjh

# 如果我们想用 SpinQuant，请参考https://github.com/XieWeikai/RotLLM/tree/dev-zjh/scripts/spinquant：
bash scripts/spinquant/train.sh /data/share/Llama-3.2-3B 8 8 8
bash scripts/spinquant/eval.sh /data/share/Llama-3.2-3B 8 8 8
# note：如果你想使用 GPTQ 量化，请使用 bash scripts/spinquant/train.sh /data/share/Llama-3.2-3B 32 8 8 对旋转矩阵进行优化，见[SpinQuant]

# 如果我们想用 Quant.npu，请参考https://github.com/XieWeikai/RotLLM/tree/dev-zjh/scripts
bash scripts/train.sh /data/share/Llama-3.2-3B-Instruct 8 8 32 32
bash scripts/eval_qat.sh /data/share/Llama-3.2-3B-Instruct 8 8 32 32（如果想测试未经联合优化的量化模型，请使用 bash scripts/eval_ptq.sh /data/share/Llama-3.2-3B-Instruct 8 8 32 32）
```




# RotLLM
This is an implementation of [SpinQuant](https://arxiv.org/abs/2405.16406) and [QuaRot](https://arxiv.org/abs/2404.00456) for different models like Qwen. We are not intented to do exactly the same things as SpinQuant and QuaRot, instead we provide a framework to customize rotation operations for any models you want to use.

## Usage
```python
# CUDA == 12.1
# Python == 3.9
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.57.1
pip install protobuf==6.32.0
pip install datasets==2.20.0
pip install accelerate==1.10.1
pip install tensorboard==2.20.0

git clone https://github.com/XieWeikai/RotLLM.git
cd RotLLM
git checkout dev-zjh
```


## Quick Adaptation to New Models

The current codebase fully supports Llama, Qwen2.5, SmolLM2, and Qwen3.
Adapting a new model is straightforward: it only requires adding a few lines of code to implement the corresponding modeling.py. Taking Llama3.2 as an example, we first reuse the MLP implementation from [Hugging Face Transformers](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/llama/modeling_llama.py):

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

我们无须修改 MLP 类型代码整体逻辑，仅需加几行代码完成 Online Rotation 即可完成自定义 MLP 类的实现。同时，我们可以直接使用代码中已提供的 apply_R4_change_model，仅需修改几处类型名即可。将该文件以其模型类型为名放入 [modeling](https://github.com/XieWeikai/RotLLM/tree/dev-zjh/modeling) 


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


 We can then directly reuse the provided apply_R4_change_model utility, with only minor modifications to the class names:
 
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



## Example
We provide a unified interface to rotate a model.
```python
import rotate
... # do whatever you want
rotate.rotate_model(model, ...) # parameters are customizable
```
You can find an example for `Qwen2ForCausalLM` and `Qwen2VLForConditionalGeneration` in [`qwen2.5-instruct.py`](./example/qwen2.5-instruct.py).

## WorkFlow of RotLLM
### Operations
The rotation operation on a model can be viewed as sequentially executing a series of predefined operations. Suppose you want to add a rotation operation for a model `abc`, first create `abc.py` in `rotate/model` and define operations as following
```python
from ..common import RotateOperationRegistry

# register the first step of operation to rotate model abc
@RotateOperationRegistry.register(abc)
def first_operation(model: abc, ...):
    ... # do whatever you want

@RotateOperationRegistry.register(abc)
def second_operation(model: abc, ...):
    ... # do whatever you want
```
After doing that, `rotate.rotate_model(model, ...)` will sequantially call `first_operation` and `second_operation` to handle model.

### Steps to rotate a model
#### Fuse layer norm
To ensure the invariance of a model, we should first fuse some operations of `norm` into the adjacent linear module.
Formally, 
```math
norm(x) = f(x) \circ w_n + b_n
```
in layer norm, we have
```math
f(x) = \frac{x-mean(x)}{\|x-mean(x)\|}
```
in RSM norm, we have
```math
f(x) = \frac{x}{\|x\|}
```
In LLMs, norm is usually followed by linear.
```math
\begin{aligned}
linear(norm(x)) &= norm(x)W_l + b_l \\
&=\left(f(x) \circ w_n + b_n \right)W_l + b_l \\
&=\left(f(x) diag(w_n) + b_n \right)W_l + b_l \\
&=f(x) \ diag(w_n)W_l + (b_nW_l + b_l)
\end{aligned}
```
This implies that $`norm(x)`$ is substitutable with $`f(x)`$. $`w_n`$ and $`b_n`$ can be fuse into linear layer
```math
\begin{aligned}
W_l &\rightarrow diag(w_n)W_l \\
b_n &\rightarrow b_nW_l + b_l
\end{aligned}
```

This is done by `fuse_layer_norms` in [rotatioin_utils.py](./rotate/rotation_utils.py).

The key problem is how `fuse_layer_norms` should identify the norm layers and their succeeding linear layers in diverse model architectures.

In our framework, to support a model like abc, you must implement a `NormLinearIterator` in abc.py, which iterates through the model and yields all `(father, norm_name, linears)` pairs. An example in [qwen.py](./rotate/model/qwen.py) is shown below
```python
from ..common import NormLinearIterator

@NormLinearIterator.register_iterator
class Qwen2NormLinearIterator(NormLinearIterator):
    def __init__(self, model: Qwen2ForCausalLM):
        super().__init__()
        self.model = model
        
    def __iter__(self):
        for layer in self.model.model.layers:
            yield layer, "input_layernorm", [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ]
            yield layer, "post_attention_layernorm", [
                layer.mlp.up_proj,
                layer.mlp.gate_proj,
            ]
        yield self.model.model, "norm", [self.model.lm_head]
        
    @classmethod
    def supports_model(cls, model: nn.Module) -> bool:
        return isinstance(model, Qwen2ForCausalLM) or isinstance(model, Qwen2VLForConditionalGeneration)
```

#### Rotate the model
The rotation operation on a model can be viewed as applying rotational transformations to either the inputs or outputs of certain layers while ensuring mathematical equivalence before and after rotation.

For different layer types (e.g., `embedding` and `linear`), the implementation of rotating their outputs varies. However, at an abstract level, both cases involve rotating outputs.

To streamline the code logic, our framework introduces the `AutoOperation` class, which encapsulates the same operation across different layers. This eliminates the need for conditional statements when applying the same operation to different layer types.

For details, you can refer to [common.py](./rotate/common.py) and [qwen.py](./rotate/model/qwen.py).

## Training rotation matrix
Currently, the rotation matrices we use are all random Hadamard matrices, which may not achieve optimal performance. According to SpinQuant, we can adopt a QAT (Quantization-Aware Training)-like approach to learn the rotation matrices for better results. This functionality has not yet been implemented and remains a TODO item.

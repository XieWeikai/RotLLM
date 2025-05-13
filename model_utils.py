from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Iterable, List
import torch.nn as nn

from transformers import Qwen2ForCausalLM, Qwen2VLForConditionalGeneration

class NormLinearIterator(ABC):
    """iterate over norm and its subsequent linear layers"""
    
    _registered_iterators: List["NormLinearIterator"] = []
    
    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[nn.Module, str, Iterable[nn.Linear]]]:
        """(parent_module, norm_layer_name, [linear_layers])"""
        pass
    
    @classmethod
    @abstractmethod
    def supports_model(cls, model: nn.Module) -> bool:
        """check if the model is supported"""
        pass
    
    @classmethod
    def register_iterator(cls, iter_cls) -> "NormLinearIterator":
        """register an iterator class"""
        cls._registered_iterators.append(iter_cls)
        return iter_cls
    
    @classmethod
    def from_model(cls, model: nn.Module) -> "NormLinearIterator":
        for iterator_cls in cls._registered_iterators:
            if iterator_cls.supports_model(model):
                return iterator_cls(model)
        
        raise ValueError(
            f"No suitable NormLinearIterator found for model type {type(model)}. "
            "Consider implementing and registering a custom iterator."
        )
        

from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
   
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
        return isinstance(model, Qwen2ForCausalLM)
    

@NormLinearIterator.register_iterator
class Qwen2ViTNormLinearIterator(NormLinearIterator):
    def __init__(self, model: Qwen2VisionTransformerPretrainedModel):
        super().__init__()
        self.model = model
        
    def __iter__(self):
        for layer in self.model.blocks:
            yield layer, "norm1", [layer.attn.qkv]
            yield layer, "norm2", [layer.mlp.fc1]
        yield self.model.merger, "ln_q", [self.model.merger.mlp[0]]
    
    
    def supports_model(cls, model: nn.Module) -> bool:
        return isinstance(model, Qwen2VisionTransformerPretrainedModel)


from typing import Dict, Type, Callable, Any, Union
import torch
import torch.nn as nn

class AutoOperationMeta(type):
    def __getattr__(cls, name):
        if name.startswith('_'):
            raise AttributeError(name)
        
        def method(module: nn.Module, *args, **kwargs):
            return cls.apply_operation(name, module, *args, **kwargs)
        
        return method

class AutoOperation(metaclass=AutoOperationMeta):
    """
    A class that supports registering and applying operations to different module types.
    Operations can be registered externally and applied to modules dynamically.
    """
    
    # Nested dictionary to store operations:
    # {operation_name: {module_type: operation_func}}
    _operations: Dict[str, Dict[Type[nn.Module], Callable]] = {}
    
    @classmethod
    def register_operation(cls, operation_name: str, module_type: Type[nn.Module]):
        """
        Decorator to register an operation for a specific module type.
        
        Args:
            operation_name: Name of the operation (e.g., 'rotate_input')
            module_type: The module type this operation applies to
        """
        def decorator(func: Callable):
            if operation_name not in cls._operations:
                cls._operations[operation_name] = {}
            cls._operations[operation_name][module_type] = func
            return func
        return decorator
    
    @classmethod
    def apply_operation(cls, operation_name: str, module: nn.Module, *args, **kwargs) -> Any:
        """
        Apply a registered operation to a module.
        
        Args:
            operation_name: Name of the operation to apply
            module: The module to apply the operation to
            *args, **kwargs: Additional arguments to pass to the operation
            
        Returns:
            The result of the operation (if any)
            
        Raises:
            ValueError: If the operation is not registered for the module type
        """
        if operation_name not in cls._operations:
            raise ValueError(f"Operation '{operation_name}' not registered")
            
        module_type = type(module)
        for base in module_type.__mro__:
            if base in cls._operations[operation_name]:
                return cls._operations[operation_name][base](module, *args, **kwargs)
        
        raise ValueError(f"Operation '{operation_name}' not registered for module type {module_type}")
    
    @classmethod
    def has_operation(cls, operation_name: str, module_type: Type[nn.Module]) -> bool:
        """
        Check if an operation is registered for a module type.
        """
        if operation_name not in cls._operations:
            return False
        return any(base in cls._operations[operation_name] for base in module_type.__mro__)
    
    # Convenience methods (dynamically generated based on registered operations)
    def __getattr__(cls, name):
        if name.startswith('_'):
            raise AttributeError(name)
        
        def method(module: nn.Module, *args, **kwargs):
            return cls.apply_operation(name, module, *args, **kwargs)
        
        return method

@AutoOperation.register_operation("rotate_input", nn.Linear)
def op_rotate_linear_input(
    linear: torch.nn.Linear,
    R: torch.Tensor):
    """
    Rotate the input of linear layers by a rotation matrix.
    i.e. xW + b -> (xR)W + b ==> x(RW) + b
    This is done by multiplying the weight matrix by the rotation matrix.
    The rotation matrix should be orthogonal.
    """
    dtype = linear.weight.dtype
    R_device = R.device
    w_device = linear.weight.device
    W_ = linear.weight.data.to(device=R_device, dtype=torch.float64)
    # note that the W_ in linear is transpose of W
    linear.weight.data = (W_ @ (R.T.to(torch.float64))).to(device=w_device, dtype=dtype)
        

@AutoOperation.register_operation("rotate_output", nn.Linear)
def op_rotate_linear_output(
    linear: nn.Linear,
    R: torch.Tensor):
    """
    Rotate the output of linear layers by a rotation matrix.
    i.e. o = xW + b -> o = (xW + b)R ==> o = x(WR) + bR
    This is done by multiplying the weight matrix by the rotation matrix.
    The rotation matrix should be orthogonal.
    """
    assert R.shape[0] == R.shape[1], "R should be a square matrix"
    assert R.shape[0] == linear.weight.shape[0], "R should be same size as output dim of linear layer"
    dtype = linear.weight.dtype
    R_device = R.device
    w_device = linear.weight.device
    W_ = linear.weight.data.to(device=R_device, dtype=torch.float64)
    # note that the W_ in linear is transpose of W
    linear.weight.data = (R.T.to(torch.float64) @ W_).to(device=w_device, dtype=dtype)
    # rotate the bias
    if linear.bias is not None:
        bias = linear.bias.data.to(device=R_device, dtype=torch.float64)
        linear.bias.data = (bias @ R.to(torch.float64)).to(device=linear.bias.device, 
                                                            dtype=linear.bias.dtype)
    
@AutoOperation.register_operation("rotate_output", nn.Embedding)
def op_rotate_embedding(
    embedding: torch.nn.Embedding,
    R: torch.Tensor):
    """
    Rotate each embedding vector by a rotation matrix R.
    """
    dtype = embedding.weight.dtype
    R_device = R.device
    w_device = embedding.weight.device
    W_ = embedding.weight.data.to(device=R_device, dtype=torch.float64)
    # note that the W_ in linear is transpose of W
    embedding.weight.data = (W_ @ (R.to(torch.float64))).to(device=w_device, dtype=dtype)

        
from transformers.models.qwen2_vl.modeling_qwen2_vl import PatchEmbed
@AutoOperation.register_operation("rotate_output", PatchEmbed)
def op_rotate_patch_embed_output(
    patch_embed: PatchEmbed,
    R: torch.Tensor):
    linear = patch_embed.proj
    assert R.shape[0] == R.shape[1], "R should be a square matrix"
    assert R.shape[0] == linear.weight.shape[0], "R should be same size as output dim of linear layer"
    dtype = linear.weight.dtype
    shape = linear.weight.shape
    R_device = R.device
    w_device = linear.weight.device
    W_ = linear.weight.data.to(device=R_device, dtype=torch.float64).view(R.shape[0], -1)
    # note that the W_ in linear is transpose of W
    linear.weight.data = (R.T.to(torch.float64) @ W_).to(device=w_device, dtype=dtype).reshape(shape)
   
"""
# denote centering the vector x as C(x) = x - mu
# we have C(x) = x - mu = x - mu 1 where 1 is the vector of ones
#  = x - 1/d sum(x) 1
# we have sum(x) = x_1 + x_2 + ... + x_n = 1^T x
# so we have C(x) = x - 1/d (1^T x) 1 = x - 1/d 1 (1^T x) = x - 1/d 1 1^T x
# that is, we can write C(x) = (I - 1/d 1 1^T) x
# denote the matrix I - 1/d 1 1^T as C
# we have C(x) = C x
# here all the vectors are column vectors
# it is easy to see that C is a symmetric matrix
# so for a row vector x we have C(x) = x C^T = x C
"""
@AutoOperation.register_operation("center_output", nn.Linear)
def op_center_linear_output(linear: torch.nn.Linear):
    """
    Center the output of linear layers
    i.e. xW + b -> (xW + b) C = xW C + bC
    that is we need to center the weight matrix by row and the bias
    """
    dtype = linear.weight.dtype
    W_ = linear.weight.data.to(dtype=torch.float64)
    # note that the W_ in linear is transpose of W
    # center echo columns of W equivalent to centering the rows of W_
    W_mean = W_.mean(dim=0, keepdim=True)
    W_centered = W_ - W_mean
    linear.weight.data = W_centered.to(dtype=dtype)
    if linear.bias is not None:
        bias = linear.bias.data.to(dtype=torch.float64)
        bias_mean = bias.mean()
        bias_centered = bias - bias_mean
        linear.bias.data = bias_centered.to(dtype=dtype)
        
@AutoOperation.register_operation("center_output", PatchEmbed)
def op_center_patch_embed_output(patch_embed: PatchEmbed):
    linear = patch_embed.proj
    dtype = linear.weight.dtype
    W_ = linear.weight.data.to(dtype=torch.float64).view(linear.weight.shape[0], -1)
    # note that the W_ in linear is transpose of W
    # center echo columns of W equivalent to centering the rows of W_
    W_mean = W_.mean(dim=0, keepdim=True)
    W_centered = W_ - W_mean
    linear.weight.data = W_centered.to(dtype=dtype).reshape(linear.weight.shape)
    


from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention

@AutoOperation.register_operation("rotate_attn_v", Qwen2Attention)
@AutoOperation.register_operation("rotate_attn_v", Qwen2VLAttention)
def op_rotate_attn_v_for_LM(
    attn: Union[Qwen2Attention, Qwen2VLAttention],
    R_v: torch.Tensor):
    """
    rotate the v (one of the inputs of attention) by a rotation matrix R_v 
    and rotate v back before W_o
    """
    config = attn.config
    num_qo_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    
    # rotate v in attention
    # i.e. rotate the output of W_v
    # note that the output is something like [v_1, v_2, ..., v_{num_heads}]
    # where v_i is a head_dim vector
    # so we need to rotate each head
    # results should be something like [v_1R_v, v_2R_v, ..., v_{num_heads}R_v]
    # this is equal to [v_1, v_2, ..., v_{num_heads}] @ diag(R_v, R_v, ..., R_v) (num_heads times)
    # so we need to rotate the output of W_v by diag(R_v, R_v, ..., R_v)
    R_v_rot = torch.block_diag(*([R_v] * num_kv_heads))
    # rotate_linear_output([attn.v_proj], R_v_rot)
    AutoOperation.rotate_output(attn.v_proj, R_v_rot)
    
    # then we need to rotate back the input of W_o
    # since o_i is linear combination of v_i
    # we can rotate the o_i by R_v^T to get back the original o_i
    # rotate_linear_input([attn.o_proj], torch.block_diag(*([R_v] * num_qo_heads)).T)
    AutoOperation.rotate_input(attn.o_proj, torch.block_diag(*([R_v] * num_qo_heads)).T)
    
from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionAttention, VisionFlashAttention2, VisionSdpaAttention

@AutoOperation.register_operation("rotate_attn_v", VisionAttention)
@AutoOperation.register_operation("rotate_attn_v", VisionFlashAttention2)
@AutoOperation.register_operation("rotate_attn_v", VisionSdpaAttention)
def op_rotate_attn_v_for_ViT(
    attn: Union[VisionAttention, VisionFlashAttention2, VisionSdpaAttention],
    R_v: torch.Tensor):
    """
    rotate the v (one of the inputs of attention) by a rotation matrix R_v 
    and rotate v back before W_o
    """
    num_heads = attn.num_heads
    dim = attn.proj.weight.shape[0]
    head_dim = dim // num_heads
    
    
    # shape of qkv.weight: [3 * dim, dim]
    q_proj, k_proj, v_proj = attn.qkv.weight.view(3, dim, dim).unbind(0) # now shape of v_proj is [dim, dim] (out_dim, in_dim)
    q_bias, k_bias, v_bias = attn.qkv.bias.view(3, dim).unbind(0)
    
    # v_proj: [dim, dim] can be view as [num_heads * head_dim, dim]
    # view it as [num_heads, head_dim, dim]
    dtype = v_proj.dtype
    device = v_proj.device
    R_device = R_v.device
    v_proj = v_proj.view(num_heads, head_dim, dim).to(device=R_device, dtype=torch.float64)
    v_proj = (R.T.unsqueeze(0).to(torch.float64) @ v_proj).to(device=device, dtype=dtype)
    v_proj = v_proj.view(dim, dim) # change it back to the original shape
    
    # rotate v_bias
    # v_bias: [dim]
    # which can be view as [num_heads, head_dim]
    v_bias = v_bias.view(num_heads, head_dim).to(dtype=torch.float64, device=R_device)
    v_bias = (v_bias @ R.to(torch.float64)).to(dtype=dtype, device=device).view(-1)
    
    # stack to get the original qkv back
    qkv = torch.stack([q_proj, k_proj, v_proj], dim=0)
    qkv = qkv.view(3 * dim, dim)
    qkv = qkv.to(device=device, dtype=dtype)
    attn.qkv.weight.data = qkv
    attn.qkv.bias.data = torch.cat([q_bias, k_bias, v_bias], dim=0)
    
    # rotate the output of W_o
    AutoOperation.rotate_input(attn.proj, torch.block_diag(*([R_v] * num_heads)).T)


if __name__ == "__main__":
    from rotation_utils import get_orthogonal_matrix
    ############# test rotate_v_for_ViT ##################
    s = 4
    dim = 512
    num_heads = 16
    head_dim = dim // num_heads
    v_attn = VisionAttention(
        dim=dim,
        num_heads=num_heads,
    )
    
    x = torch.randn(s, dim)
    R = get_orthogonal_matrix(head_dim, "hadamard")
    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)
    freqs = torch.zeros(s, head_dim // 2, dtype=torch.float32)
    x_ref = v_attn(x, cu_seqlens, freqs)
    AutoOperation.rotate_attn_v(v_attn, R)
    x_out = v_attn(x, cu_seqlens, freqs)
    assert torch.allclose(x_out, x_ref, atol=1e-5), "Output does not match expected rotated output"
    print("Output matches expected rotated output")
    ######################################################
    
    ############# test rotate_output for linear ##################
    n = 4
    in_dim = 512
    out_dim = 1024
    l = nn.Linear(in_dim, out_dim, bias=True)
    R = get_orthogonal_matrix(out_dim, "hadamard")
    
    x = torch.randn(n, in_dim)
    x_ref = l(x) @ R.to(l.weight.dtype)
    AutoOperation.rotate_output(l, R)
    x_out = l(x)
    assert torch.allclose(x_out, x_ref, atol=1e-5), "Output does not match expected rotated output"
    print("Output matches expected rotated output")
    ##############################################################
    
    ############# test rotate_output for PatchEmbed ##################
    patch_size = 14
    temporal_patch_size = 2
    in_channels = 3
    embed_dim = 1152
    
    patch_embed = PatchEmbed(
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
    )
    R = get_orthogonal_matrix(embed_dim, "hadamard")
    x = torch.randn(n, in_channels, temporal_patch_size, patch_size, patch_size)
    x_ref = patch_embed(x) @ R.to(patch_embed.proj.weight.dtype)
    AutoOperation.rotate_output(patch_embed, R)
    x_out = patch_embed(x)
    assert torch.allclose(x_out, x_ref, atol=1e-5), "Output does not match expected rotated output"
    print("Output matches expected rotated output")
    ###############################################################
    
    ######## test center_output for linear ##################
    n = 4
    in_dim = 512
    out_dim = 1024
    l = nn.Linear(in_dim, out_dim, bias=True)
    x = torch.randn(n, in_dim)
    l_x = l(x)
    x_ref = l_x - l_x.mean(dim=1, keepdim=True)
    AutoOperation.center_output(l)
    x_out = l(x)
    assert torch.allclose(x_out, x_ref, atol=1e-5), "Output does not match expected centered output"
    print("Output matches expected centered output")
    ##############################################################
    
    
    ######## test center_output for PatchEmbed ##################
    patch_size = 14
    temporal_patch_size = 2
    in_channels = 3
    embed_dim = 1152
    patch_embed = PatchEmbed(
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
    )
    x = torch.randn(n, in_channels, temporal_patch_size, patch_size, patch_size)
    l_x = patch_embed(x)
    x_ref = l_x - l_x.mean(dim=1, keepdim=True)
    AutoOperation.center_output(patch_embed)
    x_out = patch_embed(x)
    assert torch.allclose(x_out, x_ref, atol=1e-5), "Output does not match expected centered output"
    print("Output matches expected centered output")
    ##############################################################
    
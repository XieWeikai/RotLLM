import torch
import copy
import functools
import types
from tqdm import tqdm


from train.config import QuantizeConfig
from train.train_parameter import FakeQuantizer
from utils.utils import log


def copy_func_with_new_globals(f, globals=None):
    """Based on https://stackoverflow.com/a/13503277/2988730 (@unutbu)"""
    if globals is None:
        globals = f.__globals__
    g = types.FunctionType(
        f.__code__,
        globals,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=f.__closure__,
    )
    g = functools.update_wrapper(g, f)
    g.__module__ = f.__module__
    g.__kwdefaults__ = copy.copy(f.__kwdefaults__)
    return g


def add_wrapper_after_function_call_in_method(
    module,
    method_name,
    function_name,
    wrapper_fn,
):
    """
    This function adds a wrapper after the output of a function call in the method named `method_name`.
    Only calls directly in the method are affected. Calls by other functions called in the method are not affected.
    """

    original_method = getattr(module, method_name).__func__
    method_globals = dict(original_method.__globals__)
    wrapper = wrapper_fn(method_globals[function_name])
    method_globals[function_name] = wrapper
    new_method = copy_func_with_new_globals(original_method, globals=method_globals)
    setattr(module, method_name, new_method.__get__(module))
    return wrapper


class QKRotationQuantWrapper(torch.nn.Module):
    def __init__(self, func, R3, k_quant_config: QuantizeConfig, to_quant: bool = True):
        super().__init__()
        self.func = func

        self.k_quant_config = copy.deepcopy(k_quant_config)

        # Optional Rotation Matrix
        self.R3 = R3
        if to_quant:
            self.kQuant = FakeQuantizer(self.k_quant_config)
        else:
            self.kQuant = None

    def forward(self, *args, **kwargs):
        query_states, key_states = self.func(*args, **kwargs)

        # We modify (add R3)
        q_type = query_states.dtype
        k_type = key_states.dtype
        q_device = query_states.device
        k_device = key_states.device
        query_states = query_states.to(dtype = self.R3.weight.dtype) @ self.R3.weight.to(device=q_device)
        key_states = key_states.to(dtype = self.R3.weight.dtype) @ self.R3.weight.to(device=k_device)

        query_states = query_states.to(dtype=q_type)
        key_states = key_states.to(dtype=k_type)
        
        # Transpose: To unify the second dimension of the input parameter scale of StaticLearnableFakeQuantizeFunction as seqlen
        # In order to uniformly perform truncation on this dimension in StaticLearnableFakeQuantizeFunction
        key_states = key_states.transpose(1, 2)
        # Key:
        if self.kQuant is not None:
            key_states = self.kQuant(key_states)

        # Transpose again: to prevent affecting subsequent calculations
        key_states = key_states.transpose(1, 2)

        return query_states, key_states


class VQuantWrapper(torch.nn.Module):
    def __init__(self, func, config, v_quant_config: QuantizeConfig, to_quant: bool = True):
        super().__init__()
        self.func = func
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.v_quant_config = copy.deepcopy(v_quant_config)

        if to_quant:
            self.vQuant = FakeQuantizer(self.v_quant_config)
        else:
            self.vQuant = None

    def forward(self, *args, **kwargs):
        value_states = self.func(*args, **kwargs)
        bsz, q_len, _ = value_states.size()

        # 可能会根据不同模型做出对应的调整
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # Value:
        if self.vQuant is not None:
            value_states = self.vQuant(value_states)

        # reshape back to original shape (bsz, q_len, hidden_size)
        value_states = value_states.view(bsz, q_len, -1)
        return value_states

  
def add_qk_rotation_wrapper_after_function_call_in_forward(module, function_name, model_type, local_rank, *args, **kwargs):
    """
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    """
    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    # 获取 forward 方法的全局变量
    import inspect
    forward_globals = module.forward.__globals__
    
    # 如果函数不在全局变量中，尝试查找
    if function_name not in forward_globals:
        # 查找可能的函数位置
        modeling_locations = {
            "llama": "transformers.models.llama.modeling_llama",
            "qwen2": "transformers.models.qwen2.modeling_qwen2",
            "qwen3": "transformers.models.qwen3.modeling_qwen3",
        }   
        try:
            location = modeling_locations[model_type]
            module_obj = __import__(location, fromlist=[function_name])
            func = getattr(module_obj, function_name)
            forward_globals[function_name] = func
            if local_rank == 0:
                log.info(f"Found {function_name} in {location}")
        except (ImportError, AttributeError):
            raise ValueError(f"Warning: {function_name} not found in standard locations")
           
    # 现在应该能在全局变量中找到函数
    wrapper = add_wrapper_after_function_call_in_method(
        module, 
        "forward", 
        function_name, 
        functools.partial(QKRotationQuantWrapper, *args, **kwargs),
    )
    setattr(module, attr_name, wrapper)


def add_v_quant_wrapper_after_function_call_in_forward(module, function_name, *args, **kwargs):
    """
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    """
    attr_name = f"{function_name}_v_quant_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = add_wrapper_after_function_call_in_method(
        module,
        "forward",
        function_name,
        functools.partial(VQuantWrapper, *args, **kwargs),
    )
    setattr(module, attr_name, wrapper)


def add_qkv_rotation_quant(model, R3_list, k_quant_config: QuantizeConfig, v_quant_config: QuantizeConfig, to_quant: bool = True, local_rank=None):
    if local_rank is None:
        local_rank = 0
    qk_rope_function_name = "apply_rotary_pos_emb"
    v_proj_function_name = "my_Flinear"
    layers = model.model.layers
    
    for i in tqdm(range(len(layers)), desc="Wrapping apply_rotary_pos_emb and v_proj", disable=not (local_rank == 0)):
        layer = layers[i]
        add_qk_rotation_wrapper_after_function_call_in_forward(
            layer.self_attn,
            qk_rope_function_name,
            model.config.model_type,
            local_rank,
            R3=R3_list[i],
            k_quant_config=k_quant_config,
            to_quant=to_quant
        )

        add_v_quant_wrapper_after_function_call_in_forward(
            layer.self_attn.v_proj,
            v_proj_function_name,
            config=model.config,
            v_quant_config=v_quant_config,
            to_quant=to_quant
        )

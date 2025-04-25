from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Iterable, List
import torch.nn as nn

from transformers import Qwen2ForCausalLM

class NormLinearIterator(ABC):
    """iterate over norm and its subsequent linear layers"""
    
    _registered_iterators: List["NormLinearIterator"] = []
    
    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[nn.Module, Iterable[nn.Linear]]]:
        """(norm_layer, [linear_layers])"""
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
        
   
@NormLinearIterator.register_iterator
class Qwen2NormLinearIterator(NormLinearIterator):
    def __init__(self, model: Qwen2ForCausalLM):
        self.model = model
        
    def __iter__(self):
        for layer in self.model.model.layers:
            yield layer.input_layernorm, [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ]
            yield layer.post_attention_layernorm, [
                layer.mlp.up_proj,
                layer.mlp.gate_proj,
            ]
        yield self.model.model.norm, [self.model.lm_head]
        
    @classmethod
    def supports_model(cls, model: nn.Module) -> bool:
        return isinstance(model, Qwen2ForCausalLM)


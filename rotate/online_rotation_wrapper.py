import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
from rotate import AutoOperation


class MLPWrapperFactory:
    CLS_DICT = {}
    
    @staticmethod
    def register_wrapper(cls):
        def wrapper_decorator(wrapper_cls):
            if not issubclass(wrapper_cls, nn.Module):
                raise TypeError(f"{wrapper_cls.__name__} must be a subclass of nn.Module")
            MLPWrapperFactory.CLS_DICT[cls.__name__] = wrapper_cls
            print(f"Registered wrapper for {cls.__name__}")
            return wrapper_cls
        return wrapper_decorator
    
    @staticmethod
    def can_wrap(module):
        """Check if a module can be wrapped."""
        cls_name = module.__class__.__name__
        return cls_name in MLPWrapperFactory.CLS_DICT
    
    @staticmethod
    def create_wrapper(mlp, hadamard_up=None, hadamard_gate=None, hadamard_down=None):
        cls_name = mlp.__class__.__name__
        if cls_name in MLPWrapperFactory.CLS_DICT:
            return MLPWrapperFactory.CLS_DICT[cls_name](mlp, hadamard_up, hadamard_gate, hadamard_down)
        else:
            raise ValueError(f"No wrapper registered for class {cls_name}")


@MLPWrapperFactory.register_wrapper(Qwen2MLP)
class MLPWrapper(nn.Module):
    def __init__(self, mlp: Qwen2MLP, 
                 hadamard_up: torch.Tensor = None, 
                 hadamard_gate: torch.Tensor = None,
                 hadamard_down: torch.Tensor = None):
        super(MLPWrapper, self).__init__()
        self.mlp = mlp
        
        # assuming all weights are on the same device
        self.device = mlp.up_proj.weight.device
        self.dtype = mlp.up_proj.weight.dtype
        
        self.rotate_up = True if hadamard_up is not None else False
        if self.rotate_up:
            AutoOperation.rotate_output(mlp.up_proj, hadamard_up)
            self.register_buffer("hadamard_up_T", hadamard_up.T.to(self.device, dtype=self.dtype))
            
        self.rotate_gate = True if hadamard_gate is not None else False
        if self.rotate_gate:
            AutoOperation.rotate_output(mlp.gate_proj, hadamard_gate)
            self.register_buffer("hadamard_gate_T", hadamard_gate.T.to(self.device, dtype=self.dtype))
        
        self.rotate_down = True if hadamard_down is not None else False
        if self.rotate_down:
            AutoOperation.rotate_input(mlp.down_proj, hadamard_down.T)
            self.register_buffer("hadamard_down", hadamard_down.to(self.device, dtype=self.dtype))
        

    def forward(self, x):
        up = self.mlp.up_proj(x)
        gate = self.mlp.gate_proj(x)
        
        # rotate back
        if self.rotate_up:
            up = up @ self.hadamard_up_T
            
        if self.rotate_gate:
            gate = gate @ self.hadamard_gate_T
        
        gated_output = up * self.mlp.act_fn(gate)
        
        if self.rotate_down:
            # rotate
            gated_output = gated_output @ self.hadamard_down
        
        return self.mlp.down_proj(gated_output)
    

def replace_mlp(model: nn.Module,
                hadamard_generator: callable = None):
    """ 
    Replace all MLP modules in a model with MLPWrapper.
    
    Args:
        model: The model to modify
        hadamard_generator: A callable that takes module name as input and returns 
                          (hadamard_up, hadamard_gate, hadamard_down) tuple
    """
    if hadamard_generator is None:
        return
    
    # Collect modules to replace (avoid modifying dict during iteration)
    modules_to_replace = []
    
    for name, module in model.named_modules():
        if MLPWrapperFactory.can_wrap(module):
            # Generate hadamard matrices for this module
            hadamard_up, hadamard_gate, hadamard_down = hadamard_generator(name)
            
            # Only replace if at least one matrix is not None
            if hadamard_up is not None or hadamard_gate is not None or hadamard_down is not None:
                modules_to_replace.append((name, module, hadamard_up, hadamard_gate, hadamard_down))
    
    # Replace modules
    for name, original_module, hadamard_up, hadamard_gate, hadamard_down in modules_to_replace:
        # Create wrapper
        wrapper = MLPWrapperFactory.create_wrapper(
            original_module, hadamard_up, hadamard_gate, hadamard_down
        )
        
        # Replace the module in the model
        parent_module = model
        name_parts = name.split('.')
        
        # Navigate to parent module
        for part in name_parts[:-1]:
            parent_module = getattr(parent_module, part)
        
        # Replace the module
        setattr(parent_module, name_parts[-1], wrapper)
        print(f"Replaced {name} with wrapper")


if __name__ == "__main__":
    # Example usage
    from rotate import get_orthogonal_matrix
    from dataclasses import dataclass
    
    @dataclass
    class Config:
        hidden_size: int
        intermediate_size: int
        hidden_act: str = "silu"

    # Test 1: Single MLP wrapper test
    print("=" * 50)
    print("Test 1: Single MLP wrapper test")
    print("=" * 50)
    config = Config(hidden_size=512, intermediate_size=2048)
    mlp = Qwen2MLP(config)
    # Example Hadamard matrices
    hadamard_down = get_orthogonal_matrix(config.intermediate_size, mode='hadamard')
    x = torch.randn(10, 512)
    y_ref = mlp(x)
    wrapper = MLPWrapperFactory.create_wrapper(mlp, hadamard_down=hadamard_down)
    y = wrapper(x)
    print("Output matches:", torch.allclose(y, y_ref, atol=1e-5))
    
    # Test 2: replace_mlp function test
    print("\n" + "=" * 50)
    print("Test 2: replace_mlp function test")
    print("=" * 50)
    
    # Create a simple model with multiple MLP modules
    class TestModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.mlp1 = Qwen2MLP(config)
            self.mlp2 = Qwen2MLP(config)
            self.linear = nn.Linear(config.hidden_size, config.hidden_size)  # Non-MLP module
            
        def forward(self, x):
            x = self.mlp1(x)
            x = self.linear(x)
            x = self.mlp2(x)
            return x
    
    # Create test model
    test_model = TestModel(config)
    
    # Test input
    test_input = torch.randn(5, config.hidden_size)
    
    # Get reference output before replacement
    test_model.eval()
    with torch.no_grad():
        ref_output = test_model(test_input)
    
    # Define hadamard generator
    def test_hadamard_generator(module_name):
        print(f"Generating matrices for module: {module_name}")
        if "mlp1" in module_name:
            # Only apply rotation to mlp1
            hadamard_up = get_orthogonal_matrix(config.intermediate_size, mode='hadamard')
            hadamard_gate = get_orthogonal_matrix(config.intermediate_size, mode='hadamard')
            hadamard_down = get_orthogonal_matrix(config.intermediate_size, mode='hadamard')
            return hadamard_up, hadamard_gate, hadamard_down
        elif "mlp2" in module_name:
            # Only apply down rotation to mlp2
            hadamard_down = get_orthogonal_matrix(config.intermediate_size, mode='hadamard')
            return None, None, hadamard_down
        else:
            # No rotation for other modules
            return None, None, None
        
    
    # Print original model structure
    print("\nBefore replacement:")
    print(f"Model structure: {test_model}")
    
    # Apply replacement
    print("\nApplying replacement...")
    replace_mlp(test_model, test_hadamard_generator)
    
    # Print modified model structure
    print(f"After replacement: {test_model}")
    
    
    # Test output after replacement
    test_model.eval()
    with torch.no_grad():
        wrapped_output = test_model(test_input)
    
    print(f"\nOutput shape matches: {ref_output.shape == wrapped_output.shape}")
    print(f"Output values close: {torch.allclose(ref_output, wrapped_output, atol=1e-4)}")
    
    
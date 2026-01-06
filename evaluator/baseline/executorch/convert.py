import torch
from tqdm import tqdm
import copy
import importlib


from .core.qlinear import QLinear, QLinearLPBQ, QLinearW8A16_PerChannelSym
from .core.rms_norm import QRMSNorm
from .core.qdq import ActivationQDQ
from train.train_utils import untie_word_embeddings
from utils.utils import log

ActivationQDQ_to_FakeQuantizer = {
    "self_attn.q_proj_input_qdq": "self_attn.q_proj.actQuant",
    "self_attn.q_proj_output_qdq": "self_attn.q_proj.outActQuant",
    "self_attn.k_proj_output_qdq": "self_attn.k_proj.outActQuant",
    "self_attn.v_cast_to_int16_qdq": "self_attn.v_proj.outActQuant",
    # "self_attn.q_rope_add_0_output_qdq": "self_attn.apply_rotary_pos_emb_qk_rotation_wrapper.qQuant",
    # "self_attn.k_cast_to_int8_qdq": "self_attn.apply_rotary_pos_emb_qk_rotation_wrapper.kQuant",
    # "self_attn.v_cast_to_int8_qdq": "self_attn.v_proj.my_Flinear_v_quant_wrapper.vQuant",
    "self_attn.attn_value_matmul_output_qdq": "self_attn.o_proj.actQuant",
    "add_0_lhs_input_qdq": "self_attn.o_proj.outActQuant",
    "mlp.up_proj_input_qdq": "mlp.up_proj.actQuant",
    "mlp.up_proj_output_qdq": "mlp.up_proj.outActQuant",
    "mlp.gate_proj_output_qdq": "mlp.gate_proj.outActQuant",
    "mlp.down_proj_input_qdq": "mlp.down_proj.actQuant",
    "add_1_lhs_input_qdq": "mlp.down_proj.outActQuant"
}


def get_module_by_name_attr(model, name: str):
    cur = model
    for part in name.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def replace_module_by_name(model, name, new_module):
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def freeze_qwen3_rmsnorm_weight(m):
    if isinstance(m, QRMSNorm):
        m.freeze_weight()

def freeze_qwen3_linear_weight(m):
    if isinstance(m, QLinearLPBQ) or isinstance(m, QLinearW8A16_PerChannelSym):
        m.freeze_weight()

def disable_qdq_observer(m):
    if isinstance(m, ActivationQDQ):
        m.disable_observer()

def enable_qdq_observer(m):
    if isinstance(m, ActivationQDQ):
        m.enable_observer()

def disable_activation_observer(model):
    model.apply(disable_qdq_observer)

def enable_activation_observer(model):
    model.apply(enable_qdq_observer)


def disable_qdq_fakequant(m):
    if isinstance(m, ActivationQDQ):
        m.disable_fakequant()

def enable_qdq_fakequant(m):
    if isinstance(m, ActivationQDQ):
        m.enable_fakequant()

def disable_activation_fakequant(model):
    model.apply(disable_qdq_fakequant)

def enable_activation_fakequant(model):
    model.apply(enable_qdq_fakequant)


def rotllm_transform_to_executorch(model_rotllm_cuda, batch, model_args, R4, local_rank):
    device = model_rotllm_cuda.device
    model_rotllm = model_rotllm_cuda.cpu()
    del model_rotllm_cuda

    model_type = model_rotllm.config.model_type
    model_classes_name = {
        "qwen2": "Qwen2ForCausalLM",
        "qwen3": "Qwen3ForCausalLM",
    }
    model_class = None
    model_class_name = model_classes_name.get(model_type)
    if model_class_name is None:
        raise ValueError(f"Executorch: Unsupported model_type: {model_type}")
    try:
        modeling_module = importlib.import_module(f".modeling_{model_type}", package=__package__)
    except ModuleNotFoundError:
        raise ImportError(f"Cannot find modeling module for '{model_type}' (expected ./modeling_{model_type}.py)")

    model_class = getattr(modeling_module, model_class_name)
    if local_rank == 0:
        log.info(f"✅ Successfully loaded {model_class.__name__}.")
    model_executorch = model_class.from_pretrained(model_args.input_model, attn_implementation="eager").to(device=device)

    untie_word_embeddings(model_executorch)

    model_executorch.model.config.use_cache = False
    model_executorch.mllm_qualcomm_max_length = 2048
    model_executorch.eval()

    # embedding
    model_executorch.model.embed_tokens.weight.data.copy_(model_rotllm.model.embed_tokens.weight.data.to(model_executorch.model.embed_tokens.weight.device))

    # weight
    to_replace_weight_quant = []

    for name, module_exec in tqdm(model_executorch.named_modules(), desc="Rotllm transform to executorch(weight)"):
        if not isinstance(module_exec, (QLinear, QRMSNorm)):
            continue

        module_rotllm = get_module_by_name_attr(model_rotllm, name)
        assert module_rotllm is not None, f"[MISS] {name} not found in model_rotllm"
        if isinstance(module_exec, QLinear):
            assert hasattr(module_rotllm, "linear"), f"{name} has no .linear in rotllm"
            assert hasattr(module_rotllm, "weightQuant"), f"{name} has no .weightQuant in rotllm"
            assert hasattr(module_exec, "weight_quant"), f"{name} has no .weight_quant in rotllm"
            
            if "lm_head" not in name:
                to_replace_weight_quant.append((name, copy.deepcopy(module_rotllm.weightQuant)))
            module_rotllm = module_rotllm.linear

        assert module_exec.weight.shape == module_rotllm.weight.shape

        module_exec.weight.data.copy_(module_rotllm.weight.data.to(module_exec.weight.device))
        if hasattr(module_exec, "bias") and module_exec.bias is not None:
            module_exec.bias.data.copy_(module_rotllm.bias.data.to(module_exec.bias.device))

        if "q_norm" not in name and "k_norm" not in name and "norm" in name:
            module_exec.disable_fakequant()
    
    # fakequantizer: weightQuant
    for name, module_rotllm in to_replace_weight_quant:
        replace_module_by_name(model_executorch, name + ".weight_quant", module_rotllm.to(device))
    
    # R4
    for i in range(model_executorch.config.num_hidden_layers):
        model_executorch.model.layers[i].mlp.R4 = R4[i]

    # fakequantizer: actQuant
    for name, module_exec in tqdm(model_executorch.named_modules(), desc="Rotllm transform to executorch(FakeQuantizer)"):
        if not isinstance(module_exec, ActivationQDQ):
            continue

        parts = name.split(".")
        if len(parts) < 4:
            continue  # 不满足拆分要求
        
        # 拆分
        a = ".".join(parts[:3])
        b = ".".join(parts[3:])

        # b 在字典中找对应 FakeQuantizer
        fakequant_name = ActivationQDQ_to_FakeQuantizer.get(b, None)
        if not fakequant_name:
            continue  # 字典里没有对应关系
        
        rotllm_name = f"{a}.{fakequant_name}"
        module_rotllm = get_module_by_name_attr(model_rotllm, rotllm_name)
        assert module_rotllm is not None, f"[MISS] {name} not found in model_rotllm"

        module_rotllm_copy = copy.deepcopy(module_rotllm)
        replace_module_by_name(model_executorch, name, module_rotllm_copy.to(device))

    if local_rank == 0:
        log.info(f"✅ Successfully convert to executorch from RotLLM.") 

    model_executorch.apply(freeze_qwen3_rmsnorm_weight)
    model_executorch.apply(freeze_qwen3_linear_weight)
    if local_rank == 0:
        log.info(f"All PTQ weights preparation done.") 

    
    enable_activation_observer(model_executorch)
    # disable_activation_fakequant(model_executorch)

    with torch.no_grad(): 
        for i in tqdm(range(batch.size(0)), desc="Init scale and zero_point for static quant (Executorch)"):
            sample = batch[i].unsqueeze(0)  # 保持 batch 维度
            model_executorch(sample)
    if local_rank == 0:
        log.info("✅ Model_executorch init scale and zero_point ok!")

    disable_activation_observer(model_executorch)
    # enable_activation_fakequant(model_executorch)
    if local_rank == 0:
        log.info("Calibration completed, activation quantization parameters frozen.")

    return model_executorch



    

    
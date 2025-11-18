import torch
from typing import Optional
import importlib

from train.config import AllQuantizeConfigs
from train.prepare_model import (
    untie_word_embeddings, 
    replace_linear_with_rotation_quant, 
    collect_fakequant_configs, 
    model_down_proj_groupsize
)
from utils.fuse_norm_utils import fuse_layer_norms
from utils.rotation_utils import get_orthogonal_matrix
from .rotate_model import rotate_model
from .rtn import rtn_fwrd
from .gptq import gptq_fwrd
from utils.data_utils import get_wikitext2
from train.train_parameter import NoLearnRotateModule
from .static_rtn import static_rtn_fwrd
from .trainable_static_rtn import trainable_static_rtn_fwrd
from modeling.monkeypatch import add_qkv_rotation_quant
from utils.utils import log
from attention.core import Quant_scaled_dot_product_attention

def set_special_quantization_configuration(model, ptq_args):
    subset = collect_fakequant_configs(model)
    num_heads = model.config.num_attention_heads
    dim = model.config.hidden_size
    head_dim = dim // num_heads

    for name in subset:
        # weight:
        if "weightQuant" in name:
            subset[name].num_bits = 16

        # activation:
        if ptq_args.a_bits < 16 and "actQuant" in name:
            if "lm_head" in name:
                subset[name].num_bits = 16
            if "down_proj" in name:
                if ptq_args.int8_down_proj:
                    subset[name].num_bits = 8
                if ptq_args.a_groupsize > 0:
                    down_proj_groupsize = model_down_proj_groupsize(model, ptq_args.a_groupsize)
                    subset[name].groupsize = down_proj_groupsize
            if "o_proj" in name:
                subset[name].groupsize = head_dim

        # value:
        if ptq_args.v_bits < 16 and "vQuant" in name:
            if "v_proj" in name:
                subset[name].groupsize = head_dim
    return model 


def prepare_model(model, dataset, quant_configs: AllQuantizeConfigs, ptq_args, model_args, batch: Optional[torch.Tensor] = None):
    device = model.device
    model.eval()

    # untie embedding and lm_head
    untie_word_embeddings(model)
    # Model preprocessing
    fuse_layer_norms(model)

    model.config.use_cache = False

    # Prepare rotation matrix
    num_layers = model.config.num_hidden_layers
    dim = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = dim // num_heads
    hidden_dim = model.config.intermediate_size

    # Generate online rotation matrix
    R3 = [NoLearnRotateModule(get_orthogonal_matrix(head_dim, mode="hadamard", device=device)) for _ in range(num_layers)]
    R4 = [NoLearnRotateModule(get_orthogonal_matrix(hidden_dim, mode="hadamard", device=device)) for _ in range(num_layers)]  


    # 添加在线旋转矩阵 R4
    model_type = model.config.model_type
    # 动态导入对应的 modeling 模块
    try:
        modeling_module = importlib.import_module(f"modeling.{model_type}")
    except ModuleNotFoundError:
        raise ImportError(f"Cannot find modeling module for '{model_type}' (expected modeling/{model_type}.py)")

    # 检查模块中是否定义了 apply_R4_change_model
    if not hasattr(modeling_module, "apply_R4_change_model"):
        raise AttributeError(f"'modeling.{model_type}' does not define function 'apply_R4_change_model'")

    # 调用函数
    func = getattr(modeling_module, "apply_R4_change_model")
    log.info(f"✅ Found 'apply_R4_change_model' in modeling.{model_type}, now calling it...")
    func(model, R4)

    if ptq_args.trainable_R:
        assert model_args.output_rotation_path is not None, "We must give the output_rotation_path in the command line."
        R_path = model_args.output_rotation_path
        R1 = torch.load(R_path)["model.embed_tokens.R_post"].cuda().to(torch.float32)
    
        layers = [layer for layer in model.model.layers]
        R2 = []  
        for idx, layer in enumerate(layers):
            key = f"model.layers.{idx}.self_attn.v_proj.R_post"
            R_post = torch.load(R_path)[key].cuda().to(torch.float32)
            R2.append(R_post)    
    else:
        R1 = get_orthogonal_matrix(dim, mode="hadamard", device=device)
        R2 = [get_orthogonal_matrix(head_dim, mode="hadamard", device=device) for _ in range(num_layers)]   


    # Combining compatible rotation matrices
    rotate_model(model, R1, R2, [module.weight for module in R4])

    if ptq_args.mode == "dynamic":
        # Complete the calibration of GPTQ quantization, simulate the quantization of weights, and truly update the weights
        if ptq_args.w_rtn:
            # weight(RTN)
            rtn_fwrd(model, quant_configs.weight)
        else:
            # weight(GPTQ) 
            trainloader = get_wikitext2(
                dataset=dataset,
                nsamples=ptq_args.nsamples,
                seed=ptq_args.seed,
                model=model_args.input_model,
                seqlen=2048,
                eval_mode=False,
            )
            # quantize other layers with gptq
            gptq_fwrd(model, trainloader, quant_configs.weight)

    # Add all the quantizers, replacing the linear layer with RotationQuantLinear that does not contain rotation matrices
    # (with the parameter rotation_map set to None)
    model = replace_linear_with_rotation_quant(
        model,
        quant_configs=quant_configs,
    )

    # 必须要先将所有的 linear 替换成 RotationQuantLinear，然后再调用下面函数为 Value 添加量化操作，同时还对 Query、Key 添加在线旋转 R3、量化操作
    add_qkv_rotation_quant(model, R3, quant_configs.key, quant_configs.value)

    if ptq_args.sageattn:
        torch.nn.functional.scaled_dot_product_attention = Quant_scaled_dot_product_attention

    # Adjust the settings of the quantizer for the special layer, change the config, 
    # and set the weight config to 16 bits (i.e., not quantized, since it has already been quantized previously).
    if ptq_args.mode == "dynamic":
        model = set_special_quantization_configuration(model, ptq_args)
    else:
        if ptq_args.trainable_scale:
            trainable_static_rtn_fwrd(model, ptq_args, model_args) 
        else:
            static_rtn_fwrd(model, batch, ptq_args)

    return model

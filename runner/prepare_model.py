import torch
import torch.nn as nn
from typing import Optional, List
from tqdm import tqdm
import importlib


from train.config import AllQuantizeConfigs
from train.train_model import RotationQuantLinear, RotationEmbedding
from utils.fuse_norm_utils import fuse_layer_norms
from utils.rotation_utils import get_orthogonal_matrix
from train.train_parameter import LearnRotateModule, NoLearnRotateModule, FakeQuantizer
from modeling.monkeypatch import add_qkv_rotation_quant
from utils.utils import get_local_rank, log, set_config_attribute
from utils.adapt_mix_precision import adapt_modify_quantization_precision, collect_fakequant_configs
from attention.my_sdpa import Quant_scaled_dot_product_attention
from utils.adapt_online_rotation import adapt_choose_online_rotation
from evaluator.utils.rotate_model import rotate_model
from utils.data_utils import get_wikitext2
from evaluator.utils.rtn import rtn_fwrd
from evaluator.utils.gptq import gptq_fwrd
from evaluator.utils.static_rtn import static_rtn_fwrd
from evaluator.utils.trainable_static_rtn import trainable_static_rtn_fwrd
from train.train_utils import build_rotation_map, untie_word_embeddings
from evaluator.baseline.executorch.convert import rotllm_transform_to_executorch
from utils.adapt_down_16bits_input import adapt_choose_down_16bits_input


def replace_modules_with_rotation(
    model: nn.Module,
    quant_configs: AllQuantizeConfigs,
    rotation_map: dict = None,
    is_rotated: bool = True
):
    """
    Directly replace targeted nn.Linear and nn.Embedding layers in the text tower 
    with RotationQuantLinear and RotationEmbedding respectively.
    """
    if not rotation_map:
        return model

    for full_name, (R_pre, R_post, rotation_pos) in rotation_map.items():
        try:
            module = model.get_submodule(full_name)
        except AttributeError:
            print(f"Warning: Module {full_name} not found in model. Skipping.")
            continue

        if not is_rotated:
            R_pre, R_post, rotation_pos = None, None, "none"

        # Construct the corresponding new rotation-quantization module based on the type of the original module.
        new_module = None
        if isinstance(module, nn.Linear):
            new_module = RotationQuantLinear(
                config=quant_configs,
                linear=module,
                rotation_pos=rotation_pos,
                R_pre=R_pre,
                R_post=R_post
            )
        elif isinstance(module, nn.Embedding):
            new_module = RotationEmbedding(
                embedding=module,
                rotation_pos=rotation_pos,
                R_pre=R_pre,
                R_post=R_post
            )
        
        if new_module is None:
            continue

        # Parse the parent module path and the current submodule name.
        if '.' in full_name:
            parent_path, child_name = full_name.rsplit('.', 1)
            parent_module = model.get_submodule(parent_path)
        else:
            parent_module = model
            child_name = full_name
            
        setattr(parent_module, child_name, new_module)

    return model



def model_down_proj_groupsize(model, groupsize):
    assert groupsize > 1, "groupsize should be greater than 1!"

    if model.config.intermediate_size % groupsize == 0:
        return groupsize

    group_num = int(model.config.hidden_size / groupsize)
    assert groupsize * group_num == model.config.hidden_size, "Invalid groupsize for llama!"

    down_proj_groupsize = model.config.intermediate_size // group_num
    assert down_proj_groupsize * group_num == model.config.intermediate_size, "Invalid groupsize for down_proj!"
    return down_proj_groupsize


def set_special_quantization_configuration_static(model, ptq_args):
    subset = collect_fakequant_configs(model)
    num_heads = model.config.num_attention_heads
    dim = model.config.hidden_size
    head_dim = dim // num_heads

    for name in subset:
        # weight:
        if ptq_args.w_bits < 32 and "weightQuant" in name:
            if "lm_head" in name:
                subset[name].num_bits = 32
            if ptq_args.int8_down_proj and "down_proj" in name:
                subset[name].num_bits = 8

        # activation:
        if ptq_args.a_bits < 32 and "actQuant" in name:
            if "lm_head" in name:
                subset[name].num_bits = 32
            if "down_proj" in name:
                if ptq_args.int8_down_proj:
                    subset[name].num_bits = 8
                if ptq_args.a_groupsize > 0:
                    down_proj_groupsize = model_down_proj_groupsize(model, ptq_args.a_groupsize)
                    subset[name].groupsize = down_proj_groupsize
            if "o_proj" in name:
                subset[name].groupsize = head_dim

        # value:
        if ptq_args.v_bits < 32 and "vQuant" in name:
            if "v_proj" in name:
                subset[name].groupsize = head_dim

        # out_activation:
        if ptq_args.oa_bits < 32 and "outActQuant" in name:
            if "lm_head" in name:
                subset[name].num_bits = 32
    return model 



def set_special_quantization_configuration_dynamic(model, ptq_args):
    subset = collect_fakequant_configs(model)
    num_heads = model.config.num_attention_heads
    dim = model.config.hidden_size
    head_dim = dim // num_heads

    for name in subset:
        # weight:
        if "weightQuant" in name:
            if ptq_args.stage == "eval":
                subset[name].num_bits = 32
            if "lm_head" in name:
                subset[name].num_bits = 32

        # activation:
        if ptq_args.a_bits < 32 and "actQuant" in name:
            if "lm_head" in name:
                subset[name].num_bits = 32
            if "down_proj" in name:
                if ptq_args.int8_down_proj:
                    subset[name].num_bits = 8
                if ptq_args.a_groupsize > 0:
                    down_proj_groupsize = model_down_proj_groupsize(model, ptq_args.a_groupsize)
                    subset[name].groupsize = down_proj_groupsize
            # if "o_proj" in name:
            #     subset[name].groupsize = head_dim

        # # value:
        # if ptq_args.v_bits < 32 and "vQuant" in name:
        #     if "v_proj" in name:
        #         subset[name].groupsize = head_dim

        # out_activation:
        if ptq_args.oa_bits < 32 and "outActQuant" in name:
            if "lm_head" in name:
                subset[name].num_bits = 32
    return model 



def share_linear_scale_and_zero_point(model, local_rank=None):
    if local_rank is None:
        local_rank = 0
    layers = model.model.layers
    for i in tqdm(range(len(layers)), desc="Sharing parameter scale and zero_point (qkv_proj and gateup)", disable=not (local_rank == 0)):
        layer = layers[i]
        shared_parameter = {
            "qkv_scale": None,    # q_proj, k_proj, v_proj 共享
            "qkv_zero_point": None,
            "gateup_scale": None, # gate_proj, up_proj 共享
            "gateup_zero_point": None,
        }
        for name, linear in layer.named_modules():
            if isinstance(linear, RotationQuantLinear):
                # 共享 q/k/v 的激活量化 scale
                if any(key in name for key in ["q_proj", "k_proj", "v_proj"]):
                    if shared_parameter["qkv_scale"] is None:
                        shared_parameter["qkv_scale"] = linear.actQuant.scale
                        shared_parameter["qkv_zero_point"] = linear.actQuant.zero_point
                    else:
                        # 引用同一个 nn.Parameter 对象
                        linear.actQuant.scale = shared_parameter["qkv_scale"]
                        linear.actQuant.zero_point = shared_parameter["qkv_zero_point"]
                    # 由于该处 scale、zero_point 由3个 linear 共享，因此计算 LSQ+ activation initialization loss 时需要除以3，避免重复累积 loss
                    linear.actQuant.config.warmup_share_parameter_num = 3

                # 共享 gate/up 的激活量化 scale
                elif any(key in name for key in ["gate_proj", "up_proj"]):
                    if shared_parameter["gateup_scale"] is None:
                        shared_parameter["gateup_scale"] = linear.actQuant.scale
                        shared_parameter["gateup_zero_point"] = linear.actQuant.zero_point
                    else:
                        linear.actQuant.scale = shared_parameter["gateup_scale"]
                        linear.actQuant.zero_point = shared_parameter["gateup_zero_point"]
                    # 由于该处 scale、zero_point 由2个 linear 共享，因此计算 LSQ+ activation initialization loss 时需要除以2，避免重复累积 loss
                    linear.actQuant.config.warmup_share_parameter_num = 2


def prepare_model(model, dataset, quant_configs: AllQuantizeConfigs, ptq_args, model_args, batch: Optional[torch.Tensor] = None):
    device = model.device
    model.eval()
    local_rank = get_local_rank()

    # untie embedding and lm_head
    untie_word_embeddings(model)
    # Model preprocessing
    fuse_layer_norms(model)

    # Set training parameters
    for param in model.parameters():
        param.requires_grad = False

    set_config_attribute(model, "use_cache", False)

    # Prepare rotation matrix
    num_layers = model.config.num_hidden_layers
    dim = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = dim // num_heads
    hidden_dim = model.config.intermediate_size

    # Generate Hadamard rotation matrix
    R1 = LearnRotateModule(get_orthogonal_matrix(dim, mode="hadamard", device=device))
    R2 = [LearnRotateModule(get_orthogonal_matrix(head_dim, mode="hadamard", device=device)) for _ in range(num_layers)]
    
    if ptq_args.adaptive_online_rotation_R4:
        R4_hadamard = [NoLearnRotateModule(get_orthogonal_matrix(hidden_dim, mode="hadamard", device=device)) for _ in range(num_layers)]  
    else:
        R4 = [NoLearnRotateModule(get_orthogonal_matrix(hidden_dim, mode="hadamard", device=device)) for _ in range(num_layers)]  

    R3 = [NoLearnRotateModule(get_orthogonal_matrix(head_dim, mode="identity", device=device)) for _ in range(num_layers)]
    if ptq_args.adaptive_online_rotation_R4:
        R4 = [NoLearnRotateModule(get_orthogonal_matrix(hidden_dim, mode="identity", device=device)) for _ in range(num_layers)]  

    if ptq_args.adaptive_down_input_activation_16bits:
        R4 = [NoLearnRotateModule(get_orthogonal_matrix(hidden_dim, mode="identity", device=device)) for _ in range(num_layers)] 
    assert not ptq_args.adaptive_online_rotation_R4 or not ptq_args.adaptive_down_input_activation_16bits, "Adaptive 16 bits and adaptive R4 cannot be used at the same time."

    
    if ptq_args.stage == "eval":
        if ptq_args.trainable_R:
            assert model_args.output_rotation_path is not None, "We must give the output_rotation_path in the command line."
            assert not ptq_args.adaptive_online_rotation_R4 and not ptq_args.adaptive_mixed_precision and not ptq_args.adaptive_down_input_activation_16bits, "trainable_R and the adaptive strategy cannot be used at the same time."

            R_path = model_args.output_rotation_path
            R1.weight.data.copy_(
                torch.load(R_path)["model.embed_tokens.R_post"]
                .to(device=device, dtype=torch.float32)
            )
    
            layers = [layer for layer in model.model.layers]
            for idx, layer in enumerate(layers):
                key = f"model.layers.{idx}.self_attn.v_proj.R_post"
                R_post = torch.load(R_path)[key]
                R2[idx].weight.data.copy_(R_post.to(device=device, dtype=torch.float32))


            adaptive_R4 = torch.load(R_path)["adaptive_R4"]
            if len(adaptive_R4) != 0:
                for i in range(num_layers):
                    if not adaptive_R4[i]:
                        R4[i] = NoLearnRotateModule(get_orthogonal_matrix(hidden_dim, mode="identity", device=device))


    # Add the online rotation matrix R4.
    model_type = model.config.model_type
    # Dynamically import the corresponding modeling module.
    try:
        modeling_module = importlib.import_module(f"modeling.{model_type}")
    except ModuleNotFoundError:
        raise ImportError(f"Cannot find modeling module for '{model_type}' (expected modeling/{model_type}.py)")

    # Check whether `apply_R4_change_model` is defined in the module.
    if not hasattr(modeling_module, "apply_R4_change_model"):
        raise AttributeError(f"'modeling.{model_type}' does not define function 'apply_R4_change_model'")

    # Call the function.
    func = getattr(modeling_module, "apply_R4_change_model")
    if local_rank == 0:
        log.info(f"✅ Found 'apply_R4_change_model' in modeling.{model_type}, now calling it...")
    func(model, R4, local_rank)


    if ptq_args.stage == "eval":
        # Combining compatible rotation matrices
        rotate_model(model, R1.weight, [module.weight for module in R2], [module.weight for module in R4])

        # target_dtype = torch.bfloat16 
        # model.to(dtype=target_dtype)
        # model.config.dtype = target_dtype
        # model.config.text_config.dtype = target_dtype        
        # model.save_pretrained("/data/share/Qwen3-VL-2B-Instruct-rotated-test-zjh", safe_serialization=True)

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
             
        # Prepare the rotation matrix and the rotation position
        rotation_map = build_rotation_map(model, R1, R2, R4)

        # Add all the quantizers, replacing the linear layer with RotationQuantLinear that does not contain rotation matrices
        # Because `rotate_model` has already rotated the weights.
        model = replace_modules_with_rotation(
            model, 
            quant_configs=quant_configs, 
            rotation_map=rotation_map,
            is_rotated=False
        )
    else:
        # Prepare the rotation matrix and the rotation position
        rotation_map = build_rotation_map(model, R1, R2, R4)

        # Call the replacement function, replace the linear layer and embedding layer, and add the rotation matrix and quantizer
        model = replace_modules_with_rotation(
            model, 
            quant_configs=quant_configs, 
            rotation_map=rotation_map,
            is_rotated=True
        )

    # All linear layers must first be replaced with `RotationQuantLinear`. 
    # Then the following function is called to add quantization operations to the Value, 
    # while also applying online rotation R3 and quantization to the Query and Key.

    add_qkv_rotation_quant(model, R3, quant_configs.query, quant_configs.key, quant_configs.value, local_rank=local_rank)
    # if ptq_args.q_bits < 32:
    #     torch.nn.functional.scaled_dot_product_attention = Quant_scaled_dot_product_attention

    if ptq_args.mode == "dynamic":
        model = set_special_quantization_configuration_dynamic(model, ptq_args)
    else:
        # Adjust the settings of the quantizer for special layers, change the config.
        model = set_special_quantization_configuration_static(model, ptq_args) 


    if ptq_args.stage == "train" or not ptq_args.trainable_R:
        # By default, each layer requires the online rotation matrix R4.
        adaptive_R4 = {i: True for i in range(num_layers)}
        # Initialize all quantizers using the calibration set.
        if quant_configs.weight.mode == "static":
            assert batch is not None, "We need to prepare the initial sample set required for static quantization."
            model.eval()

            if ptq_args.adaptive_mixed_precision or ptq_args.adaptive_online_rotation_R4 or ptq_args.adaptive_down_input_activation_16bits:
                batch_find_threshold = batch[-ptq_args.adapt_need_sample:]
                batch = batch[:ptq_args.need_sample_for_static_init]

                if ptq_args.adaptive_online_rotation_R4 or ptq_args.adaptive_down_input_activation_16bits:
                    # 将 donw_proj 层的 activation 修改为 8 bit
                    from utils.adapt_online_rotation import modify_down_proj_activation_8bit
                    model = modify_down_proj_activation_8bit(model)

            if local_rank == 0:
                collect_fakequant_configs(model, "./txt/one.txt", True)

            with torch.no_grad(): 
                for i in tqdm(range(batch.size(0)), desc="Init scale and zero_point for static quant", disable=not (local_rank == 0)):
                    sample = batch[i].unsqueeze(0)  # 保持 batch 维度
                    model(sample)
                if local_rank == 0:
                    log.info(f"✅ Init scale and zero_point ok!")

            if ptq_args.adaptive_mixed_precision or ptq_args.adaptive_online_rotation_R4 or ptq_args.adaptive_down_input_activation_16bits:
                assert batch_find_threshold is not None, "batch_find_threshold should not be empty."
                
                if ptq_args.adaptive_online_rotation_R4:
                    # 选择在线旋转矩阵 R4
                    model, adaptive_R4 = adapt_choose_online_rotation(model, R4_hadamard, batch, batch_find_threshold, ptq_args, local_rank)

                if ptq_args.adaptive_down_input_activation_16bits:
                    model = adapt_choose_down_16bits_input(model, batch, batch_find_threshold, ptq_args, local_rank)
                    adaptive_R4 = {i: False for i in range(num_layers)}

                if ptq_args.adaptive_mixed_precision:
                    model = adapt_modify_quantization_precision(model, adaptive_R4, batch, batch_find_threshold, ptq_args, local_rank)
    
    if local_rank == 0:
        log.info(model)
    
    if ptq_args.stage == "train":
        # 将 q_proj、k_proj、v_proj 前的激活量化器中的 scale、zero_point 共享同一个参数
        # 将 gate_proj、up_proj 前的激活量化器中的 scale、zero_point 共享同一个参数
        if ptq_args.mode == "static":
            share_linear_scale_and_zero_point(model, local_rank)

        # Integration of trainable parameters
        R_trainable_parameters = [R1.weight] + [r.weight for r in R2]
        q_trainable_parameters = [
            p for p in model.parameters() if p.requires_grad
        ]

        new_q_trainable_parameters = []
        for p in q_trainable_parameters:
            if all(p is not r for r in R_trainable_parameters):
                new_q_trainable_parameters.append(p)
        q_trainable_parameters = new_q_trainable_parameters


        # R_dict = {}
        # for name, module in model.named_modules():
        #     if isinstance(module, FakeQuantizer):
        #         if hasattr(module, "scale"):
        #             R_dict[f"{name}.scale"] = module.scale
        #         if hasattr(module, "zero_point"):
        #             R_dict[f"{name}.zero_point"] = module.zero_point
        
        # with open("./txt/scale_init.txt", "w") as f:
        #     for k, v in R_dict.items():
        #         if v is None:
        #             f.write(f"{k}: None\n")
        #         else:
        #             # 标量 scale / zero_point
        #             f.write(f"{k}: {v.detach().cpu().item()}\n")

        return model, adaptive_R4, R_trainable_parameters, q_trainable_parameters
    else:
        # Adjust the settings of the quantizer for the special layer, change the config, 
        # and set the weight config to 32 bits (i.e., not quantized, since it has already been quantized previously).
        if ptq_args.mode == "static":
            if ptq_args.trainable_scale:
                trainable_static_rtn_fwrd(model, batch, model_args) 
            else:
                static_rtn_fwrd(model)
        
            if ptq_args.executorch:
                model = rotllm_transform_to_executorch(model, batch, model_args, R4, local_rank)

        return model, None, None, None
    
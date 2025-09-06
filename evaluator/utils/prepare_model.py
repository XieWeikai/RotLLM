import transformers

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
# from .test_gptq import gptq_fwrd
from .gptq import gptq_fwrd
from utils.data_utils import get_wikitext2
from train.train_parameter import NoLearnRotateModule

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


def prepare_model(model, dataset, quant_configs: AllQuantizeConfigs, ptq_args, model_args):
    transformers.set_seed(ptq_args.seed)
    device = model.device

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

    # TODO: 保存模型的时候如何处理
    # Generate online rotation matrix
    R3 = [NoLearnRotateModule(get_orthogonal_matrix(head_dim, mode="hadamard", device=device)) for _ in range(num_layers)]
    R4 = [NoLearnRotateModule(get_orthogonal_matrix(hidden_dim, mode="hadamard", device=device)) for _ in range(num_layers)]  

    # Add online rotation matrices R3 and R4, but do not add the quantizer for Key and Value; 
    # wait to add this quantizer after GPTQ quantizes the weights.
    if model.config.model_type == "llama": 
        from modeling.llama import apply_R3R4_change_model 
        from modeling.llama import value_kv_quantizers 
        print("from modeling.llama import apply_R3R4_change_model")
        print("from modeling.llama import value_kv_quantizers")
    elif model.config.model_type == "qwen2": 
        from modeling.qwen2 import apply_R3R4_change_model
        from modeling.qwen2 import value_kv_quantizers 
        print("from modeling.qwen2 import apply_R3R4_change_model")
        print("from modeling.qwen2 import value_kv_quantizers")
    else:
        raise NotImplementedError(f"Unsupported model type {model.config.model_type}")

    apply_R3R4_change_model(model, R3, R4, quant_configs.key, quant_configs.value, to_quant = False)

    # Combining compatible rotation matrices
    rotate_model(model, ptq_args, [module.weight for module in R4])

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
        # gptq_fwrd(model, trainloader, "cuda", ptq_args)
        gptq_fwrd(model, trainloader, quant_configs.weight)

    # Add quantizer for Key and Value
    value_kv_quantizers(model)

    # Add all the quantizers, replacing the linear layer with RotationQuantLinear that does not contain rotation matrices
    # (with the parameter rotation_map set to None)
    model = replace_linear_with_rotation_quant(
        model,
        quant_configs=quant_configs,
    )

    # Adjust the settings of the quantizer for the special layer, change the config, 
    # and set the weight config to 16 bits (i.e., not quantized, since it has already been quantized previously).
    model = set_special_quantization_configuration(model, ptq_args)

    return model

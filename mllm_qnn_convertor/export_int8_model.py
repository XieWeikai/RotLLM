import argparse
from transformers import Qwen2VLForConditionalGeneration
import torch
import json

from utils.get_input_output_scales import get_clip_and_scale


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    w = w.to("cuda")
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_()

    if n_bits == 8:
        w = w.to("cpu").type(torch.int8)
    elif n_bits == 16 or n_bits == 32:
        w = w.to("cpu").type(torch.int32)
    else:
        w = w.to("cpu").type(torch.int8)
    scale = scales.to("cpu").type(torch.float32)
    return w, scale


@torch.no_grad()
def quantize_bias_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    w = w.to("cuda")
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_()

    # NOTE: all convert to int 32
    w = w.to("cpu").type(torch.int32)

    scale = scales.to("cpu").type(torch.float32)
    return w, scale


if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument(
        "--model_type",
        choices=["llama", "qwen1", "qwen2", "qwen2vl", "gemma", "phi", "opt", "mixtral", "falcon"],
        default="llama",
    )
    parser.add_argument("--scale_file", type=argparse.FileType("r"))
    parser.add_argument("--t01m_clip_threshold", type=int, default=64)
    parser.add_argument("--output_model", type=str, default="model-int8.mllm")
    parser.add_argument("--quant_bias", type=str, default="False")
    parser.add_argument('--quantize_vit', action='store_true', help='quantize vit or not')
    parser.add_argument('--rotate_vit', action='store_true', help='rotate vit or not')
    parser.add_argument('--online_rotate', action='store_true', help='do online rotation')
    parser.add_argument("--clip_all", action="store_true", help="clip all layer")
    args = parser.parse_args()

    print("model: ", args.model_name)
    print("model type: ", args.model_type)
    print("scale file: ", args.scale_file.name)
    print("t01m clip threshold: ", args.t01m_clip_threshold)
    print("output model: ", args.output_model)
    print("Quantize bias: ", args.quant_bias)
    print("quantize_vit: ", args.quantize_vit)
    print("rotate_vit: ", args.rotate_vit)
    print("online_rotate: ", args.online_rotate)
    print("clip_all: ", args.clip_all)


    model_name = args.model_name
    act_dict = args.scale_file.name
    t01m_clip_threshold = args.t01m_clip_threshold

    if args.model_type != "qwen2vl":
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32)
    else:
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )


    if args.online_rotate:
        R_bin = torch.load("./R.bin")
        R = R_bin["R"]
        R_v = R_bin["R_v"]
        R_vit = R_bin["R_vit"]
        R_vs_vit = R_bin["R_vs_vit"]

        from rotate import rotate_model

        rotate_model(model, R, R_v)

        # if args.rotate_vit:

        #     rotate_model(model.visual, R_vit, R_vs_vit)
    print(f"finish online rotation")

    
    act_dict = json.load(open(act_dict))

    no_clip_input = {
                # "visual.blocks.22.mlp.fc2",
                # "visual.blocks.23.mlp.fc2",
                # "visual.blocks.24.mlp.fc2",
                # "visual.blocks.25.mlp.fc2",
                # "visual.blocks.26.mlp.fc2",
                # "visual.blocks.27.mlp.fc2",
            }
            
    no_clip_output = {
        # "visual.blocks.22.mlp.fc2",
        # "visual.blocks.23.mlp.fc2",
        # "visual.blocks.24.mlp.fc2",
        # "visual.blocks.25.mlp.fc2",
        # "visual.blocks.26.mlp.fc2",
        # "visual.blocks.27.mlp.fc2",
    }

    act_scales, clip_top, return_dict = get_clip_and_scale(
        act_dict, t01m_clip_threshold, args.clip_all, no_clip_input=no_clip_input, no_clip_output=no_clip_output
    )

    print(f"clip input num: {return_dict['clip_input_num']}")
    print(f"clip output num: {return_dict['clip_output_num']}")
    print(f"no clip input num: {return_dict['no_clip_input_num']}")
    for i in return_dict["no_clip_input_name"]:
        print(f"no clip input: {i}")
    print(f"no clip output num: {return_dict['no_clip_output_num']}")
    for i in return_dict["no_clip_output_name"]:
        print(f"no clip output: {i}")


    model_dict = model.state_dict()

    for i in act_scales:
        model_dict[i + ".input_scale"] = torch.tensor(act_scales[i]["input"])
        model_dict[i + ".output_scale"] = torch.tensor(act_scales[i]["output"])
        print(i, " input scale: ", act_scales[i]["input"], " output scale: ", act_scales[i]["output"])
        model_dict[i + ".clip_input"] = torch.tensor(clip_top[i]["input"])
        model_dict[i + ".clip_output"] = torch.tensor(clip_top[i]["output"])

    new_model = {}
    for name, param in model_dict.items():
        print(name)
        # NOTE: skip visual tower
        if "vision_tower" in name:
            continue

        if "lm_head" in name or  "merger" in name:
            new_model[name] = param
            continue
        if name.replace(".weight", "") in act_scales:
            if "head" not in name:
                layer_name = name
                new_model[layer_name], scale = quantize_weight_per_tensor_absmax(
                    model_dict[layer_name], 8
                )
                new_model[layer_name + ".scale"] = scale

                # NOTE: the int8 weight used for QNN in mllm needs to be transposed
                new_model[name] = new_model[name].transpose(-2, -1)
                print(f"Quantized {layer_name} with scale {scale}")
            else:
                new_model[name] = param
                # print(f"Copy {name}")
        elif name.replace(".bias", "") in act_scales:
            if "head" not in name:
                # NOTE: k proj bias's scale is very large, so we use 32 bit int
                # if ".0.self_attn.k_proj" in name:
                #     layer_name = name
                #     new_model[layer_name], scale = quantize_bias_per_tensor_absmax(
                #         model_dict[layer_name], 32
                #     )
                #     new_model[layer_name + ".scale"] = scale
                # else:
                layer_name = name
                if args.quant_bias == "False":
                    new_model[name] = param
                    print(f"FP {layer_name}")
                else:
                    new_model[layer_name], scale = quantize_bias_per_tensor_absmax(
                        model_dict[layer_name], 8
                    )
                # if scale > 1:
                #     new_model[layer_name], scale = quantize_bias_per_tensor_absmax(
                #         model_dict[layer_name], 12
                #     )
                #     print("use new scale: ", scale)

                    new_model[layer_name + ".scale"] = scale
                        
                    print(f"Quantized {layer_name} with scale {scale}")
            else:
                new_model[name] = param
                # print(f"Copy {name}")
        else:
            new_model[name] = param
            # print(f"Copy {name}")

    torch.save(new_model, args.output_model)
    print(f"Model saved to {args.output_model}")

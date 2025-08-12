from transformers import AutoModelForCausalLM
import torch

device = "cuda:0" # specify the device to use
# dtype = "float32" # specify the data type
dtype = torch.float32

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/RotLLM/LLM/phi-3-mini-4k-instruct", device_map=device, torch_dtype=dtype)
    print(model)

    # for name, module in model.named_modules():
    #     print(name, module.__class__.__name__)


    model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/RotLLM/LLM/Qwen2.5-1.5B-Instruct", device_map=device, torch_dtype=dtype)
    print(model)

    # for name, module in model.named_modules():
    #     print(name, module.__class__.__name__)

    print("hahahahahahahaha")
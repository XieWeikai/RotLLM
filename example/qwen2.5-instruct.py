# -*- coding: utf-8 -*-

# This file shows how to rotate Qwen models using the package rotate.

import rotate
from fakequant.linear import replace_linear_with_fakequant
from transformers import AutoModelForCausalLM, AutoTokenizer

from fakequant.cache import FakeQuantDynamicCache

model_path = "/data/share/Qwen2.5-1.5B-Instruct" # specify the path to the model

device = "cuda:0" # specify the device to use
dtype = "float32" # specify the data type
MLP_ONLINE_ROTATION = True # whether to use online rotation for MLPs

def chat(tokenizer, model, prompt, max_new_tokens=1024, kv_quantization=False):
    chats = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    prompt_text = tokenizer.apply_chat_template(chats, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        past_key_values=FakeQuantDynamicCache() if kv_quantization else None
    )
    response = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
    return response


import torch
from torch import nn
from fakequant.linear import replace_linear_with_fakequant     

if __name__ == "__main__":
    # load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=dtype)
    model.eval()

    prompt = "write me a binary search in C"
    start_time_1 = time.time()
    response = chat(tokenizer, model, prompt)
    end_time_1 = time.time()
    print(response)
    
    
    
    # model info    
    num_layers = model.config.num_hidden_layers
    dim = model.config.hidden_size      
    qo_heads = model.config.num_attention_heads
    head_dim = dim // qo_heads
    intermediate_size = model.config.intermediate_size
    
    # get randome hadamard rotation matrix
    R = rotate.get_orthogonal_matrix(dim, mode="hadamard", device=device)
    R_v = [rotate.get_orthogonal_matrix(head_dim, mode="hadamard", device=device) for _ in range(num_layers)]
    # rotate the model using the rotation matrix
    # currently only supports Qwen2ForCausalLM and Qwen2VLForConditionalGeneration
    rotate.rotate_model(model, R, R_v)
    
    if MLP_ONLINE_ROTATION:
        from rotate.online_rotation_wrapper import replace_mlp
        def hadamard_generator(module_name):
            # you can customize the rotation matrices for different MLP modules here
            # for example, you can use different modes or different sizes
            hadamard_up = None
            hadamard_gate = None
            hadamard_down = rotate.get_orthogonal_matrix(intermediate_size, mode='hadamard', device=device)
            return hadamard_up, hadamard_gate, hadamard_down
        
        # replace MLP modules with online rotation wrappers
        print("Replacing MLP modules with online rotation wrappers...")
        replace_mlp(model, hadamard_generator)
        print("Replacement done.")
    
    # quantize the model using fakequant
    print("Quantizing the model using fakequant...")
    model = replace_linear_with_fakequant(model, num_bits=8)
    print("Quantization done.")
    
    # test the rotated model
    print("--------------------------------------")

    response = chat(tokenizer, model, prompt, kv_quantization=True)

    print(response)
    
    # test the rotated model with fake quantization
    # if you want to use fake quantization, uncomment the following lines
   
    
    
    # now you can save the rotated model
    
    # model.save_pretrained(model_path + "_rotated")
    # tokenizer.save_pretrained(model_path + "_rotated")
    # print(f"Rotated model saved to {model_path}_rotated")
    
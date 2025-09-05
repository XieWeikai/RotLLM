# -*- coding: utf-8 -*-

import rotate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

model_path = "/root/autodl-tmp/RotLLM/LLM/phi-3-mini-4k-instruct"  
device = "cuda:0"
dtype = torch.float32

def chat(tokenizer, model, prompt, max_new_tokens=1024):
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
    )
    response = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
    return response

from transformers import Phi3ForCausalLM
from fakequant.linear import replace_linear_with_fakequant  

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=dtype)
    model.eval()

    prompt = "write me a binary search in C"
    start_time_1 = time.time()
    response = chat(tokenizer, model, prompt)
    end_time_1 = time.time()
    print(response)

    num_layers = model.config.num_hidden_layers
    dim = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = dim // num_heads

    # 生成 Hadamard 旋转矩阵
    R = rotate.get_orthogonal_matrix(dim, mode="hadamard", device=device)
    R_v = [rotate.get_orthogonal_matrix(head_dim, mode="hadamard", device=device) for _ in range(num_layers)]

    # 调用注册的 Phi3 旋转接口旋转模型
    rotate.rotate_model(model, R, R_v)

    start_time_2 = time.time()
    response = chat(tokenizer, model, prompt)
    end_time_2 = time.time()
    print(response)

    # 如果需要假量化，取消注释下面代码
    replace_linear_with_fakequant(model, 8)
    print(f"Quantized model: {model}")
    start_time_3 = time.time()
    response = chat(tokenizer, model, prompt)
    end_time_3 = time.time()
    print(response)

    print(f"Time for first generation (original model): {end_time_1 - start_time_1:.3f} seconds\n")
    print(f"Time for third generation (rotated model): {end_time_2 - start_time_2:.3f} seconds\n") 
    print(f"Time for fourth generation (rotated + quantized model): {end_time_3 - start_time_3:.3f} seconds\n")
    

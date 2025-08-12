# -*- coding: utf-8 -*-

import rotate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

model_path = "/root/autodl-tmp/RotLLM/LLM/phi-3-mini-4k-instruct"  # 指定 Phi3 模型路径
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

if __name__ == "__main__":
    print("haha")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=dtype)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数：{total_params:,} 个")  
    model_size_MB = total_params * 4 / (1024**2)  
    print(f"模型大小约为：{model_size_MB:.2f} MB（float32）")

    print("========================================================")

    prompt = "write me a binary search in C"
    print("=== Original model response ===")
    start_time_1 = time.time()
    response = chat(tokenizer, model, prompt)
    end_time_1 = time.time()
    print(response)

    print("========================================================")

    # 如果需要假量化，取消注释下面代码
    from fakequant.linear import replace_linear_with_fakequant
    replace_linear_with_fakequant(model, 8)

    # 保存模型（保存一次后即可注释下面代码）
    model.save_pretrained(model_path + "_no_rotated_q")
    tokenizer.save_pretrained(model_path + "_no_rotated_q")
    print(f"Rotated model saved to {model_path}_no_rotated_q")

    print("=== Only quant model response ===")
    start_time_4 = time.time()
    response = chat(tokenizer, model, prompt)
    end_time_4 = time.time()
    print(response)

    model.cpu()
    del model
    torch.cuda.empty_cache()

    print("========================================================")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=dtype)
    model.eval()

    num_layers = model.config.num_hidden_layers
    dim = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = dim // num_heads

    # 生成 Hadamard 旋转矩阵
    R = rotate.get_orthogonal_matrix(dim, mode="hadamard", device=device)
    R_v = [rotate.get_orthogonal_matrix(head_dim, mode="hadamard", device=device) for _ in range(num_layers)]

    # 调用注册的 Phi3 旋转接口旋转模型
    rotate.rotate_model(model, R, R_v)

    print("=== Rotated model response ===")
    start_time_2 = time.time()
    response = chat(tokenizer, model, prompt)
    end_time_2 = time.time()
    print(response)

    # 保存模型（保存一次后即可注释下面代码）
    model.save_pretrained(model_path + "_rotated_no_q")
    tokenizer.save_pretrained(model_path + "_rotated_no_q")
    print(f"Rotated model saved to {model_path}_rotated_no_q")

    print("========================================================")

    # 如果需要假量化，取消注释下面代码
    replace_linear_with_fakequant(model, 8)
    print(f"Quantized model: {model}")
    start_time_3 = time.time()
    response = chat(tokenizer, model, prompt)
    end_time_3 = time.time()
    print("=== Quantized model response ===")
    print(response)

    # 保存模型（保存一次后即可注释下面代码）
    model.save_pretrained(model_path + "_rotated_q")
    tokenizer.save_pretrained(model_path + "_rotated_q")
    print(f"Rotated model saved to {model_path}_rotated_q")

    print("========================================================")

    print(f"Time for first generation (original model): {end_time_1 - start_time_1:.3f} seconds\n")
    print(f"Time for third generation (only quantized model): {end_time_4 - start_time_4:.3f} seconds\n")
    print(f"Time for second generation (rotated model): {end_time_2 - start_time_2:.3f} seconds\n") 
    print(f"Time for third generation (rotated + quantized model): {end_time_3 - start_time_3:.3f} seconds\n")
    

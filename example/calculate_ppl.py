import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.nn import functional as F
from tqdm import tqdm
import time

# ======== PPL =========
@torch.no_grad()
def evaluate_ppl(model, tokenizer, dataset, max_length=512, device='cuda'):
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_tokens = 0

    for example in tqdm(dataset, desc="Evaluating"):
        text = example['text']
        encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
        input_ids = encodings.input_ids.to(device)

        if input_ids.size(1) < 2:
            continue  

        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        num_tokens = input_ids.size(1)

        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return ppl

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



from transformers import AutoTokenizer
from datasets import load_dataset

MODEL_PATH_ORIG = "/root/autodl-tmp/RotLLM/LLM/phi-3-mini-4k-instruct"  # 原始模型
MODEL_PATH_QUANT = "/root/autodl-tmp/RotLLM/LLM/phi-3-mini-4k-instruct_no_rotated_q"  # 直接量化的模型
MODEL_PATH_ROT = "/root/autodl-tmp/RotLLM/LLM/phi-3-mini-4k-instruct_rotated_no_q"  # 旋转无量化的模型
MODEL_PATH_ROT_QUANT = "/root/autodl-tmp/RotLLM/LLM/phi-3-mini-4k-instruct_rotated_q"  # 旋转后再量化的模型

device = "cuda:0"
dtype = torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_ORIG)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
dataset = dataset.filter(lambda example: len(example['text'].strip()) > 0)  # 去除空行

prompt = "write me a binary search in C"

# === 评估原始模型 ===
print("========================================================")
model_orig = AutoModelForCausalLM.from_pretrained(MODEL_PATH_ORIG, device_map=device, torch_dtype=dtype)
ppl_orig = evaluate_ppl(model_orig, tokenizer, dataset)
print(f"原始模型 PPL: {ppl_orig:.2f}")

print("=== Original model response ===")
start_time_1 = time.time()
response = chat(tokenizer, model_orig, prompt)
end_time_1 = time.time()
print(response)

model_orig.cpu()
del model_orig
torch.cuda.empty_cache()

# === 评估直接量化的模型 ===
print("========================================================")
model_quant = AutoModelForCausalLM.from_pretrained(MODEL_PATH_QUANT, device_map=device, torch_dtype=dtype)
ppl_quant = evaluate_ppl(model_quant, tokenizer, dataset)
print(f"直接量化模型 PPL: {ppl_quant:.2f}")

print("=== Only quant model response ===")
start_time_4 = time.time()
response = chat(tokenizer, model_quant, prompt)
end_time_4 = time.time()
print(response)

model_quant.cpu()
del model_quant
torch.cuda.empty_cache()

# === 评估直接旋转无量化的模型 ===
print("========================================================")
model_rot = AutoModelForCausalLM.from_pretrained(MODEL_PATH_ROT, device_map=device, torch_dtype=dtype)
ppl_rot = evaluate_ppl(model_rot, tokenizer, dataset)
print(f"直接旋转无量化模型 PPL: {ppl_rot:.2f}") 

print("=== Rotated model response ===")
start_time_2 = time.time()
response = chat(tokenizer, model_rot, prompt)
end_time_2 = time.time()
print(response)

model_rot.cpu()
del model_rot
torch.cuda.empty_cache()

# === 评估旋转后再量化的模型 ===
print("========================================================")
model_rot_quant = AutoModelForCausalLM.from_pretrained(MODEL_PATH_ROT_QUANT, device_map=device, torch_dtype=dtype)
ppl_rot_quant = evaluate_ppl(model_rot_quant, tokenizer, dataset)
print(f"旋转后量化模型 PPL: {ppl_rot_quant:.2f}")

print("=== Rotated Quantized model response ===")
start_time_3 = time.time()
response = chat(tokenizer, model_rot_quant, prompt)
end_time_3 = time.time()
print(response)

model_rot_quant.cpu()
del model_rot_quant
torch.cuda.empty_cache()

print(f"Time for first generation (original model): {end_time_1 - start_time_1:.3f} seconds\n")
print(f"Time for second generation (only quantized model): {end_time_4 - start_time_4:.3f} seconds\n")
print(f"Time for third generation (rotated model): {end_time_2 - start_time_2:.3f} seconds\n") 
print(f"Time for fourth generation (rotated + quantized model): {end_time_3 - start_time_3:.3f} seconds\n")

"""
原始模型 PPL: 11.02
直接量化模型 PPL: 155847.25
直接旋转无量化模型 PPL:11.02
旋转后量化模型 PPL: 57944.80

直接量化的模型输出固定
旋转后再量化的模型输出不固定，但基本符合预期
涉及到量化的模型的输出都是有问题的，不能很好的符合我的要求

Time for first generation (original model): 6.987 seconds

Time for third generation (only quantized model): 45.397 seconds

Time for second generation (rotated model): 18.509 seconds

Time for third generation (rotated + quantized model): 61.775 seconds
"""

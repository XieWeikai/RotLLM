import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.nn import functional as F
from tqdm import tqdm

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

# === 评估原始模型 ===
model_orig = AutoModelForCausalLM.from_pretrained(MODEL_PATH_ORIG, device_map=device, torch_dtype=dtype)
ppl_orig = evaluate_ppl(model_orig, tokenizer, dataset)
print(f"原始模型 PPL: {ppl_orig:.2f}")

model_orig.cpu()
del model_orig
torch.cuda.empty_cache()

# === 评估直接量化的模型 ===
model_quant = AutoModelForCausalLM.from_pretrained(MODEL_PATH_QUANT, device_map=device, torch_dtype=dtype)
ppl_quant = evaluate_ppl(model_quant, tokenizer, dataset)
print(f"直接量化模型 PPL: {ppl_quant:.2f}")

model_quant.cpu()
del model_quant
torch.cuda.empty_cache()

# === 评估直接旋转无量化的模型 ===
model_rot = AutoModelForCausalLM.from_pretrained(MODEL_PATH_ROT, device_map=device, torch_dtype=dtype)
ppl_rot = evaluate_ppl(model_rot, tokenizer, dataset)
print(f"直接旋转无量化模型 PPL: {ppl_rot:.2f}")

model_rot.cpu()
del model_rot
torch.cuda.empty_cache()

# === 评估旋转后再量化的模型 ===
model_rot_quant = AutoModelForCausalLM.from_pretrained(MODEL_PATH_ROT_QUANT, device_map=device, torch_dtype=dtype)
ppl_rot_quant = evaluate_ppl(model_rot_quant, tokenizer, dataset)
print(f"旋转后量化模型 PPL: {ppl_rot_quant:.2f}")

model_rot_quant.cpu()
del model_rot_quant
torch.cuda.empty_cache()

"""
原始模型 PPL: 11.02
直接量化模型 PPL: 155847.25
直接旋转无量化模型 PPL:11.02
旋转后量化模型 PPL: 57944.80

直接量化的模型输出固定
旋转后再量化的模型输出不固定，但基本符合预期
涉及到量化的模型的输出都是有问题的，不能很好的符合我的要求
"""

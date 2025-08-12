from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math
from tqdm import tqdm
# import re

from fakequant.linear import replace_linear_with_fakequant
import rotate

# @torch.no_grad()
# def evaluate_piqa(model, tokenizer, dataset, device):
#     model.eval()
#     model.to(device)

#     correct = 0
#     total = 0
#     for example in tqdm(dataset, desc="Evaluating"):
#         goal = example['goal']
#         sol1 = example['sol1']
#         sol2 = example['sol2']
#         label = example['label'] + 1  # 0或1

#         prompt = (
#             f"Question: {goal}\n"
#             f"Option 1: {sol1}\n"
#             f"Option 2: {sol2}\n"
#             "Answer with only the number of the correct option: 1 or 2.\n"
#             "Only output the number."
#         )

#         inputs = tokenizer(prompt, return_tensors="pt").to(device)
#         generation_output = model.generate(
#             **inputs,
#             max_new_tokens=5,
#             do_sample=False, 
#             temperature=None,
#             top_p=None,
#             top_k=None,
#             eos_token_id=tokenizer.eos_token_id
#         )
#         output_text = tokenizer.decode(generation_output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

#         print(output_text)

#         match = re.search(r"\b[12]\b", output_text)
#         if match:
#             pred = int(match.group())
#         else:
#             pred = None
#         if pred == label:
#             correct += 1
#         total += 1

#     return correct/total

@torch.no_grad()
def evaluate_piqa_acc(model, tokenizer, dataset, device='cuda', max_length=512):
    model.eval()
    model.to(device)
    
    total = 0
    correct = 0
    
    for example in tqdm(dataset, desc="Evaluating PIQA"):
        goal = example['goal']
        sol1 = example['sol1']
        sol2 = example['sol2']
        label = example['label']  # 0或1，表示sol1或sol2是正确的
        
        text1 = f"Question: {goal} Solution 1: {sol1}"
        text2 = f"Question: {goal} Solution 2: {sol2}"
        
        def get_logprob(text):
            inputs = tokenizer(text, return_tensors='pt', 
                             truncation=True, max_length=max_length).to(device)
            outputs = model(**inputs, labels=inputs.input_ids)
            return -outputs.loss.item()
        
        logprob1 = get_logprob(text1)
        logprob2 = get_logprob(text2)
        
        # 选择概率更高的选项作为预测
        pred = 0 if logprob1 > logprob2 else 1
        
        if pred == label:
            correct += 1
        total += 1
    
    acc = correct / total if total > 0 else 0.0
    return acc


def main():
    # MODEL_PATH_ORIG = "/home/zjh/project1_LLM_Rotation_QAT/RotLLM/LLM/Qwen2.5-1.5B-Instruct"
    MODEL_PATH_ORIG = "/home/zjh/project1_LLM_Rotation_QAT/RotLLM/LLM/phi-3-mini-4k-instruct"
    device = "cuda:2"
    dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_ORIG)

    dataset = load_dataset("piqa", split="validation", trust_remote_code=True)

    dataset = dataset.select(range(10))  # ✅ 只测试前 10 条

    model_orig = AutoModelForCausalLM.from_pretrained(MODEL_PATH_ORIG, device_map=device, torch_dtype=dtype)
    model_orig.eval()

    num_layers = model_orig.config.num_hidden_layers
    dim = model_orig.config.hidden_size
    num_heads = model_orig.config.num_attention_heads
    head_dim = dim // num_heads

    # 生成 Hadamard 旋转矩阵
    R = rotate.get_orthogonal_matrix(dim, mode="hadamard", device=device)
    R_v = [rotate.get_orthogonal_matrix(head_dim, mode="hadamard", device=device) for _ in range(num_layers)]

    # 调用注册的 Phi3 旋转接口旋转模型
    rotate.rotate_model(model_orig, R, R_v)

    replace_linear_with_fakequant(model_orig, 8)

    acc = evaluate_piqa_acc(model_orig, tokenizer, dataset, device)
    print(f"旋转后量化 phi-3-mini-4k-instruct 模型 PIQA accuracy: {acc * 100:.2f}%")

    model_orig.cpu()
    del model_orig
    torch.cuda.empty_cache()

    return acc

if __name__ == "__main__":
    main()
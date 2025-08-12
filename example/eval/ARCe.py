import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F
import random

from fakequant.linear import replace_linear_with_fakequant
import rotate

@torch.no_grad()
def evaluate_arc_e_acc(model, tokenizer, dataset, device='cuda', max_length=512):
    # model.eval()
    # model.to(device)

    # total = 0
    # correct = 0

    # for example in tqdm(dataset, desc="Evaluating ARC-e (Random)"):
    #     choices = example['choices']['text']
    #     answer_key = example['answerKey']  # 例如 'A', 'B', ...

    #     option_letters = ['A', 'B', 'C', 'D'][:len(choices)]

    #     # 随机选一个选项
    #     pred_letter = random.choice(option_letters)

    #     if pred_letter == answer_key:
    #         correct += 1
    #     total += 1

    # acc = correct / total if total > 0 else 0.0
    # return acc


    model.eval()
    model.to(device)

    total = 0
    correct = 0

    for example in tqdm(dataset, desc="Evaluating ARC-e"):
        question = example['question']
        choices = example['choices']['text']
        answer_key = example['answerKey']  # 例如 'A', 'B', ...

        # 将选项映射到字母（A, B, C, D）
        option_letters = ['A', 'B', 'C', 'D'][:len(choices)]
        option_map = {letter: choice for letter, choice in zip(option_letters, choices)}

        def get_logprob(option_text):

            input_text = f"Question: {question} Option: {option_text}"
            inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=max_length).to(device)
            outputs = model(**inputs, labels=inputs.input_ids)
            return -outputs.loss.item()  

        logprobs = {}
        for letter, option_text in option_map.items():
            logprobs[letter] = get_logprob(option_text)

        pred_letter = max(logprobs.keys(), key=lambda k: logprobs[k])

        if pred_letter == answer_key:
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

    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")

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

    acc = evaluate_arc_e_acc(model_orig, tokenizer, dataset, device=device)
    print(f"旋转后量化 phi-3-mini-4k-instruct 模型 ARC-e acc: {acc * 100:.2f}%")


    model_orig.cpu()
    del model_orig
    torch.cuda.empty_cache()

    return acc

if __name__ == "__main__":
    main()
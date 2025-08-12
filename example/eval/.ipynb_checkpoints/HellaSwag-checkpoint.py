import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

from fakequant.linear import replace_linear_with_fakequant
import rotate

@torch.no_grad()
def evaluate_hellaswag_acc(model, tokenizer, dataset, device='cuda', max_length=512):
    model.eval()
    model.to(device)
    
    total = 0
    correct = 0
    
    for example in tqdm(dataset, desc="Evaluating HellaSwag"):
        ctx = example['ctx']
        endings = example['endings']
        label = int(example['label'])  # 正确答案索引（0-3）
        
        # 构造4个完整输入文本
        texts = [ctx + " " + endings[i] for i in range(4)]
        
        logprobs = []
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(device)
            # 计算整个序列的负对数概率（与loss不同）
            outputs = model(**inputs, labels=inputs.input_ids)
            logprob = -outputs.loss.item()
            logprobs.append(logprob)
        
        pred = np.argmax(logprobs)
        
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

    dataset = load_dataset("hellaswag", split="validation")

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

    acc = evaluate_hellaswag_acc(model_orig, tokenizer, dataset, device=device)
    print(f"旋转后量化 phi-3-mini-4k-instruct 模型 HellaSwag acc: {acc * 100:.2f}%")


    model_orig.cpu()
    del model_orig
    torch.cuda.empty_cache()

    return acc

if __name__ == "__main__":
    main()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F

from fakequant.linear import replace_linear_with_fakequant
import rotate

@torch.no_grad()
def evaluate_winogrande_acc(model, tokenizer, dataset, device='cuda', max_length=512):
    model.eval()
    model.to(device)

    total = 0
    correct = 0

    for example in tqdm(dataset, desc="Evaluating WinoGrande"):
        # print(example)
        sentence = example['sentence']
        option1 = example['option1']
        option2 = example['option2']
        gold = int(example['answer']) - 1

        text1 = sentence.replace("_", option1)
        text2 = sentence.replace("_", option2)

        def get_logprob(text):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(device)
            outputs = model(**inputs, labels=inputs.input_ids)
            return -outputs.loss.item()

        logprob1 = get_logprob(text1)
        logprob2 = get_logprob(text2)

        # 选择对数概率更高的选项
        pred = 0 if logprob1 > logprob2 else 1

        if pred == gold:
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

    dataset = load_dataset("winogrande", "winogrande_xl", split="validation")

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

    acc = evaluate_winogrande_acc(model_orig, tokenizer, dataset, device=device)
    print(f"旋转后量化 phi-3-mini-4k-instruct 模型 WinoGrande acc: {acc * 100:.2f}%")


    model_orig.cpu()
    del model_orig
    torch.cuda.empty_cache()

    return acc

if __name__ == "__main__":
    main()
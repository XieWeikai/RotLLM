import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.nn import functional as F
from tqdm import tqdm

from fakequant.linear import replace_linear_with_fakequant
import rotate

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


def main():
    # MODEL_PATH_ORIG = "/home/zjh/project1_LLM_Rotation_QAT/RotLLM/LLM/Qwen2.5-1.5B-Instruct"
    MODEL_PATH_ORIG = "/home/zjh/project1_LLM_Rotation_QAT/RotLLM/LLM/phi-3-mini-4k-instruct"

    device = "cuda:2"
    dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_ORIG)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset = dataset.filter(lambda example: len(example['text'].strip()) > 0)  # 去除空行

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

    ppl_rot_quant = evaluate_ppl(model_orig, tokenizer, dataset)
    print(f"旋转后量化 phi-3-mini-4k-instruct 模型 PPL: {ppl_rot_quant:.2f}")

    model_orig.cpu()
    del model_orig
    torch.cuda.empty_cache()

    return ppl_rot_quant

if __name__ == "__main__":
    main()
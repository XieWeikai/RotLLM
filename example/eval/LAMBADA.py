import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

from fakequant.linear import replace_linear_with_fakequant
import rotate

@torch.no_grad()
def evaluate_lambada_acc(model, tokenizer, dataset, device='cuda', max_length=512):
    model.eval()
    model.to(device)

    total = 0
    correct = 0

    for example in tqdm(dataset, desc="Evaluating LAMBADA"):
        # print(example)
        full_text = example["text"].rstrip()
        words = full_text.split()
        if len(words) < 2:
            continue

        target_word = words[-1]
        context = " ".join(words[:-1])

        # if not context.endswith(" "):
            # context += " "

        target_tokens = tokenizer.encode(" " + target_word, add_special_tokens=False)
        if not target_tokens:
            continue

        inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=False).to(device)
        input_ids = inputs.input_ids

        generated_ids = []
   
        # 自回归预测 target_tokens
        for _ in range(len(target_tokens)):
            outputs = model(input_ids)
            next_token_id = torch.argmax(outputs.logits[0, -1]).item()
            generated_ids.append(next_token_id)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)

        # token 级别匹配
        # print(generated_ids) 
        # print(target_tokens)
        if generated_ids == target_tokens:
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

    dataset = load_dataset("lambada", split="validation")

    dataset = dataset.select(range(5))  # ✅ 只测试前 10 条  

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

    acc = evaluate_lambada_acc(model_orig, tokenizer, dataset, device=device)
    print(f"旋转后量化 phi-3-mini-4k-instruct 模型 LAMBADA acc: {acc * 100:.2f}%")

    model_orig.cpu()
    del model_orig
    torch.cuda.empty_cache()

    return acc 

if __name__ == "__main__":
    main()
    # from transformers import AutoTokenizer

    # MODEL_PATH_ORIG = "/home/zjh/project1_LLM_Rotation_QAT/RotLLM/LLM/Qwen2.5-1.5B-Instruct"

    # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_ORIG)

    # print(f"token id 19145 对应的token是: {tokenizer.decode([19145], clean_up_tokenization_spaces=False)}")
    # print(f"token id 1008 对应的token是: {tokenizer.decode([1008], clean_up_tokenization_spaces=False)}")
    # print(f"token id 3355 对应的token是: {tokenizer.decode([3355], clean_up_tokenization_spaces=False)}")
    # print(f"token id 323 对应的token是: {tokenizer.decode([323], clean_up_tokenization_spaces=False)}")
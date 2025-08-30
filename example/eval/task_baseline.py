import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.nn import functional as F
from tqdm import tqdm
import traceback
import numpy as np

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



@torch.no_grad()
def evaluate_arc_e_acc(model, tokenizer, dataset, device='cuda', max_length=512):
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


@torch.no_grad()
def evaluate_arc_c_acc(model, tokenizer, dataset, device='cuda', max_length=512):
    model.eval()
    model.to(device)

    total = 0
    correct = 0

    for example in tqdm(dataset, desc="Evaluating ARC-c"):
        question = example['question']
        choices = example['choices']['text']
        answer_key = example['answerKey'] 

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

        # 注意下面这行代码，对于 Qwen2.5-1.5B-Instruct 模型需要修改成：target_tokens = tokenizer.encode(" " + target_word, add_special_tokens=False)
        # 对于 phi-3-mini-4k-instruct 模型需要修改成：target_tokens = tokenizer.encode(target_word, add_special_tokens=False)
        # 对于 Llama-3.2-3B-Instruct 模型需要修改成：target_tokens = tokenizer.encode(" " + target_word, add_special_tokens=False)
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
    # MODEL_PATH_ORIG = "/data/share/Qwen2.5-1.5B-Instruct"
    # MODEL_PATH_ORIG = "/data/zjh/LLM/phi-3-mini-4k-instruct"
    MODEL_PATH_ORIG = "/data/share/Llama-3.2-3B-Instruct"

    device = "cuda:2"
    dtype = torch.float32

    # 准备模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_ORIG)
    model_orig = AutoModelForCausalLM.from_pretrained(MODEL_PATH_ORIG, device_map=device, torch_dtype=dtype)
    model_orig.eval()

    # 准备所有需要的数据集
    dataset_ppl = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset_ppl = dataset_ppl.filter(lambda example: len(example['text'].strip()) > 0)  # 去除空行


    dataset_piqa = load_dataset("piqa", split="validation", trust_remote_code=True)

    dataset_winogrande = load_dataset("winogrande", "winogrande_xl", split="validation")


    dataset_hellaswag = load_dataset("hellaswag", split="validation")

    dataset_arce = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")

    dataset_arcc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")

    dataset_lambada = load_dataset("lambada", split="validation")

    """
    num_layers = model_orig.config.num_hidden_layers
    dim = model_orig.config.hidden_size
    num_heads = model_orig.config.num_attention_heads
    head_dim = dim // num_heads

    # 生成 Hadamard 旋转矩阵
    R = rotate.get_orthogonal_matrix(dim, mode="hadamard", device=device)
    R_v = [rotate.get_orthogonal_matrix(head_dim, mode="hadamard", device=device) for _ in range(num_layers)]

    rotate.rotate_model(model_orig, R, R_v)

    replace_linear_with_fakequant(model_orig, 8)
    """

    # 所有测试任务
    eval_tasks = {
        "PPL": (evaluate_ppl, dataset_ppl),
        "PIQA": (evaluate_piqa_acc, dataset_piqa),
        "WinoGrande": (evaluate_winogrande_acc, dataset_winogrande),
        "HellaSwag": (evaluate_hellaswag_acc, dataset_hellaswag),
        "ARC-e": (evaluate_arc_e_acc, dataset_arce),
        "ARC-c": (evaluate_arc_c_acc, dataset_arcc),
        "LAMBADA": (evaluate_lambada_acc, dataset_lambada),
    }

    results = {}

    result_file_path = "/home/zjh/project1_LLM_Rotation_QAT/RotLLM/example/eval/test_results_llama_fp16.txt"

    with open(result_file_path, "w", encoding="utf-8") as f:
        f.write("=== Test Results ===\n")

    for task_name, (eval_fn, dataset) in eval_tasks.items():
        print(f"Running {task_name} evaluation...")
        try:
            # dataset = dataset.select(range(5))  # ✅ 只测试前 10 条
            result = eval_fn(model_orig, tokenizer, dataset, device=device)
            results[task_name] = result
            print(f"{task_name} result: {result}")
 
            with open(result_file_path, "a", encoding="utf-8") as f:
                f.write(f"{task_name}: {result}\n")
        except Exception as e:
            err_trace = traceback.format_exc()
            print(f"Error running {task_name}: {e}")
            with open(result_file_path, "a", encoding="utf-8") as f:
                f.write(f"{task_name} ERROR: {err_trace}\n")

    model_orig.cpu()
    del model_orig
    torch.cuda.empty_cache()

    print("\n===== Summary =====")
    for k, v in results.items():
        print(f"{k}: {v}")

    return results

if __name__ == "__main__":
    main()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizerFast, Qwen2TokenizerFast
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from logging import Logger
import torch.nn.functional as F
import re


from utils.utils import get_logger


log: Logger = get_logger("RotLLM")

test = False
count = 5

@torch.no_grad()
def loglikelihood(model, tokenizer, context, continuation):
    device = model.device
    # Encode the context and options sections
    context_enc = tokenizer.encode(context, add_special_tokens=False)
    continuation_enc = tokenizer.encode(continuation, add_special_tokens=False)
    continuation_enc_len = len(continuation_enc)

    inp = torch.tensor((context_enc + continuation_enc)[-(tokenizer.model_max_length + 1) :][:-1], dtype=torch.long).to(device)
    # Plan 2: padding
    # (inplen,) = inp.shape
    # padding_length = tokenizer.model_max_length
    # pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    # inp = torch.cat([inp, torch.full((padding_length - inplen,), pad_id, dtype=inp.dtype).to(inp.device)], dim=0)
    inp = inp.unsqueeze(0) 

    logits = model(inp)[0]

    logits = F.log_softmax(logits, dim=-1)
    # Plan 2: padding
    # logits = logits[:, inplen - continuation_enc_len : inplen] # [1, len(continuation_enc), vocab]
    # Plan 1: Truncate
    logits = logits[:, -continuation_enc_len:] # [1, len(continuation_enc), vocab]

    pred_continuation_enc = logits.argmax(dim=-1)

    continuation_enc = torch.tensor(continuation_enc, device=device).unsqueeze(0)   # [1, len(continuation_enc)]
    logits = torch.gather(logits, 2, continuation_enc.unsqueeze(-1)).squeeze(-1)  # [1, seq]

    equal = (pred_continuation_enc == continuation_enc).all()

    return float(logits.sum()) / continuation_enc_len, equal



@torch.no_grad()
def evaluate_piqa_acc(model, tokenizer):
    dataset = load_dataset("piqa", split="validation", trust_remote_code=True)
    if test:
        dataset = dataset.select(range(count))  # ✅ Only test the first count items
    
    total = 0
    correct = 0
    
    for example in tqdm(dataset, desc="Evaluating PIQA"):

        goal = example['goal']
        sol1 = example['sol1']
        sol2 = example['sol2']
        label = example['label']  # 0 or 1, indicating whether sol1 or sol2 is correct
        
        context = f"Question: {goal} \nAnswer: "
        
        logprob1, _ = loglikelihood(model, tokenizer, context, sol1)
        logprob2, _ = loglikelihood(model, tokenizer, context, sol2)
        
        # Choose the option with the higher log probability
        pred = 0 if logprob1 > logprob2 else 1
        
        if pred == label:
            correct += 1
        total += 1
    
    acc = correct / total if total > 0 else 0.0
    return acc


@torch.no_grad()
def evaluate_winogrande_acc(model, tokenizer):
    dataset = load_dataset("winogrande", "winogrande_xl", split="validation")
    # dataset = load_dataset("winogrande", split="validation")
    if test:
        dataset = dataset.select(range(count))  # ✅ Only test the first count items

    total = 0
    correct = 0

    for example in tqdm(dataset, desc="Evaluating WinoGrande"):
        # print(example)
        sentence = example['sentence']
        option1 = example['option1']
        option2 = example['option2']
        gold = int(example['answer']) - 1

        pronoun_loc = sentence.index("_")
        context1 = sentence[:pronoun_loc] + option1
        context2 = sentence[:pronoun_loc] + option2

        target = sentence[pronoun_loc + 1:].strip()
        target = " " + target

        logprob1, _ = loglikelihood(model, tokenizer, context1, target)
        logprob2, _ = loglikelihood(model, tokenizer, context2, target)

        # Choose the option with the higher log probability
        pred = 0 if logprob1 > logprob2 else 1

        if pred == gold:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    return acc


@torch.no_grad()
def evaluate_hellaswag_acc(model, tokenizer):
    dataset = load_dataset("hellaswag", split="validation")
    if test:
        dataset = dataset.select(range(count))  # ✅ Only test the first count items
    
    total = 0
    correct = 0

    def preprocess(text):
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text
    
    for example in tqdm(dataset, desc="Evaluating HellaSwag"):
        activity_label = example['activity_label']
        ctx_a = example['ctx_a']
        ctx_b = example['ctx_b']
        endings = example['endings']
        label = int(example['label'])  # Correct answer index (0-3)

        ctx = ctx_a + " " + ctx_b.capitalize()
        context = preprocess(activity_label + ": " + ctx)
        targets = [" " + preprocess(ending) for ending in endings]
        
        logprobs = []
        for target in targets:
            logprob, _ = loglikelihood(model, tokenizer, context, target)
            logprobs.append(logprob)
        
        pred = np.argmax(logprobs)
        
        if pred == label:
            correct += 1
        total += 1
    
    acc = correct / total if total > 0 else 0.0
    return acc


@torch.no_grad()
def evaluate_arc_e_acc(model, tokenizer):
    dataset = load_dataset("ai2_arc", "ARC-Easy", split="test")
    if test:
        dataset = dataset.select(range(count))  # ✅ Only test the first count items

    total = 0
    correct = 0
    num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}

    for example in tqdm(dataset, desc="Evaluating ARC-e"):
        question = example['question']
        choices = example['choices']['text']
        answer_key = example['answerKey']  # For example: 'A', 'B', ...

        answer_key = num_to_letter.get(answer_key, answer_key)

        context = "Question: " + question + "\nAnswer: "

        logprobs = []
        for choice in choices:
            logprob, _ = loglikelihood(model, tokenizer, context, choice)
            logprobs.append(logprob)
        
        pred = np.argmax(logprobs)
        
        if chr(ord('A') + pred) == answer_key:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    return acc


@torch.no_grad()
def evaluate_arc_c_acc(model, tokenizer):
    dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")
    if test:
        dataset = dataset.select(range(count))  # ✅ Only test the first count items

    total = 0
    correct = 0
    num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}

    for example in tqdm(dataset, desc="Evaluating ARC-c"):
        question = example['question']
        choices = example['choices']['text']
        answer_key = example['answerKey']  # For example: 'A', 'B', ...

        answer_key = num_to_letter.get(answer_key, answer_key)

        context = "Question: " + question + "\nAnswer: "

        logprobs = []
        for choice in choices:
            logprob, _ = loglikelihood(model, tokenizer, context, choice)
            logprobs.append(logprob)
        
        pred = np.argmax(logprobs)
        
        if chr(ord('A') + pred) == answer_key:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    return acc


@torch.no_grad()
def evaluate_lambada_acc(model, tokenizer):
    dataset = load_dataset("lambada", split="test")
    if test:
        dataset = dataset.select(range(count))  # ✅ Only test the first count items

    total = 0
    correct = 0

    for example in tqdm(dataset, desc="Evaluating LAMBADA"):
        full_text = example["text"]
        
        context = full_text.rsplit(" ", 1)[0]
        target = " " + full_text.rsplit(" ", 1)[1]

        _, euqal = loglikelihood(model, tokenizer, context, target)

        if euqal:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    return acc


def task_baseline(model, tokenizer):
    # # MODEL_PATH = "/data/share/Llama-3.2-3B"
    # # MODEL_PATH = "/data/share/Llama-3.2-3B-Instruct"
    # # MODEL_PATH = "/data/share/Qwen2.5-3B-Instruct"
    # MODEL_PATH = "/data/share/SmolLM2-1.7B-Instruct"

    # device = "cuda:2"
    # dtype = torch.float16

    # # Prepare the model
    # tokenizer = AutoTokenizer.from_pretrained(
    #     pretrained_model_name_or_path=MODEL_PATH,
    #     model_max_length=2048,
    #     padding_side="right",
    #     add_eos_token=False,
    #     add_bos_token=False,
    # )
    # # tokenizer = LlamaTokenizerFast.from_pretrained(
    # #     pretrained_model_name_or_path=MODEL_PATH,
    # #     model_max_length=2048,
    # #     padding_side="right",
    # #     use_fast=True,
    # #     add_eos_token=False,
    # #     add_bos_token=False,
    # # )
    # # tokenizer = Qwen2TokenizerFast.from_pretrained(
    # #     pretrained_model_name_or_path=MODEL_PATH,
    # #     model_max_length=2048,
    # #     padding_side="right",
    # #     use_fast=True,
    # #     add_eos_token=False,
    # #     add_bos_token=False,
    # # )
    # model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=MODEL_PATH, torch_dtype=dtype).to(device=device)
    model.eval()
    model.config.use_cache = True

    # All test tasks
    eval_tasks = {
        "PIQA": evaluate_piqa_acc,
        "WinoGrande": evaluate_winogrande_acc,
        "HellaSwag": evaluate_hellaswag_acc,
        "ARC-e": evaluate_arc_e_acc,
        "ARC-c": evaluate_arc_c_acc,
        "LAMBADA": evaluate_lambada_acc,
    }

    results = {}

    # with open(result_file_path, "w", encoding="utf-8") as f:
    #     f.write("=== Results ===\n")

    for task_name, eval_fn in eval_tasks.items():
        print(f"Running {task_name} evaluation...")
        result = eval_fn(model, tokenizer)
        results[task_name] = result
        print(f"{task_name} result: {result}")

        # with open(result_file_path, "a", encoding="utf-8") as f:
        #     f.write(f"{task_name}: {result}\n")

    model.cpu()
    del model
    torch.cuda.empty_cache()

    print("\n===== Summary =====")
    for k, v in results.items():
        print(f"{k}: {v}")

    return results


if __name__ == "__main__":
    task_baseline()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from logging import Logger

from utils.data_utils import get_wikitext2
from evaluator.utils.prepare_model import prepare_model
from utils.process_args import process_args_ptq
from utils.utils import get_logger
from evaluator.utils.evaluator import evaluator

log: Logger = get_logger("RotLLM")

def eval() -> None:
    model_args, training_args, ptq_args, quant_configs = process_args_ptq()
    device = "cuda"
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16

    # TODO: (Fast)tokenizer params    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        add_eos_token=False,
        add_bos_token=False,
    )

    log.info(f"Complete tokenizer loading...")

    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_args.input_model, torch_dtype=dtype).to(device=device)

    model.config.use_cache = False
    # Prepare the dataset (for calibration and evaluation)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    
    # Prepare the model
    model = prepare_model(model, dataset, quant_configs, ptq_args, model_args)

    log.info("Model init completed for evaling...")
    log.info("Start to eval...")
    
    testloader = get_wikitext2(
        dataset,
        seed=ptq_args.seed,
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=True,
    )
    dataset_ppl = evaluator(model, testloader, training_args.model_max_length, ptq_args)
    log.info("wiki2 ppl is: {}".format(dataset_ppl))
    
if __name__ == "__main__":
    eval()

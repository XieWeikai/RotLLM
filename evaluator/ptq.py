import torch
from transformers import LlamaTokenizerFast, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from logging import Logger
import transformers

from utils.data_utils import get_wikitext2
from evaluator.utils.prepare_model import prepare_model
from utils.process_args import process_args_ptq
from utils.utils import get_logger
from evaluator.utils.evaluator import evaluator
from utils.data_utils import CustomJsonDataset

log: Logger = get_logger("RotLLM")

def eval() -> None:
    model_args, training_args, ptq_args, quant_configs = process_args_ptq()
    transformers.set_seed(ptq_args.seed)
    device = "cuda"
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16

    # TODO: (Fast)tokenizer params    
    # tokenizer = AutoTokenizer.from_pretrained(
    #     pretrained_model_name_or_path=model_args.input_model,
    #     cache_dir=training_args.cache_dir,
    #     model_max_length=training_args.model_max_length,
    #     padding_side="right",
    #     add_eos_token=False,
    #     add_bos_token=False,
    # )

    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )

    log.info(f"Complete tokenizer loading...")

    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_args.input_model, torch_dtype=dtype).to(device=device)

    model.config.use_cache = False
    # Prepare the dataset (for calibration and evaluation)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")


    train_data = CustomJsonDataset(
        dataset["train"],
        tokenizer,
        block_size=min(training_args.model_max_length, 2048),
    )

    # Prepare the calibration set for static quantization, used to initialize scale and zero_point
    num_samples = quant_configs.activation.need_sample_for_static_init
    samples = [train_data[i + 10]["input_ids"] for i in range(num_samples)]
    batch = torch.tensor(samples).to(device=model.device)
    
    # Prepare the model
    model = prepare_model(model, dataset, batch, quant_configs, ptq_args, model_args)

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

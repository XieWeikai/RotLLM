import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import transformers

from utils.data_utils import get_wikitext2
from evaluator.utils.prepare_model import prepare_model
from utils.process_args import process_args_ptq
from utils.utils import log
from evaluator.utils.evaluator import evaluator
from utils.data_utils import CustomJsonDataset
from .task import task_baseline

def eval() -> None:
    model_args, training_args, ptq_args, quant_configs = process_args_ptq()
    transformers.set_seed(ptq_args.seed)
    device = "cuda"
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_args.input_model, torch_dtype=dtype).to(device=device)

    tokenizer_classes = {
        "llama": "LlamaTokenizerFast",
        "qwen2": "Qwen2TokenizerFast",
    }
    tokenizer = None
    tokenizer_class_name = tokenizer_classes.get(model.config.model_type)
    
    if tokenizer_class_name is not None:
        try:
            tokenizer_class = getattr(__import__('transformers'), tokenizer_class_name)
            log.info(f"Attempting to use {tokenizer_class.__name__}.")
            tokenizer = tokenizer_class.from_pretrained( 
                pretrained_model_name_or_path=model_args.input_model,
                cache_dir=training_args.cache_dir,              
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=True,
                add_eos_token=False,
                add_bos_token=False,
            )
            log.info(f"✅ Successfully loaded {tokenizer_class.__name__}.")
        except Exception as e:
            log.warning(f"Failed to load {tokenizer_class_name}: {e}")
            tokenizer = None
    
    # 如果加载 Fast tokenizer 失败，则回退到 AutoTokenizer
    if tokenizer is None:
        log.info("✅ Using AutoTokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            add_eos_token=False,
            add_bos_token=False,
        )
    log.info(f"Complete tokenizer loading...")

    
    model.config.use_cache = False
    # Prepare the dataset (for calibration and evaluation)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    batch = None
    if ptq_args.mode == "static" and not ptq_args.trainable_scale:
        train_data = CustomJsonDataset(
            dataset["train"],
            tokenizer,
            block_size=min(training_args.model_max_length, 2048),
        )
        # Prepare the calibration set for static quantization, used to initialize scale and zero_point
        num_samples = ptq_args.need_sample_for_static_init
        samples = [train_data[i + 10]["input_ids"] for i in range(num_samples)]
        batch = torch.tensor(samples).to(device=model.device)
    
    # Prepare the model
    model = prepare_model(model, dataset, quant_configs, ptq_args, model_args, batch)
    # from attention.core import Quant_scaled_dot_product_attention
    # torch.nn.functional.scaled_dot_product_attention = Quant_scaled_dot_product_attention

    log.info("Model init completed for evaling...")
    log.info("💡Start to eval...")
    
    if not ptq_args.task:
        testloader = get_wikitext2(
            dataset,
            seed=ptq_args.seed,
            seqlen=2048,
            tokenizer=tokenizer,
            eval_mode=True,
        )
        dataset_ppl = evaluator(model, testloader, training_args.model_max_length, ptq_args)
        log.info("wiki2 ppl is: {}".format(dataset_ppl))
    else:
        log.info("Calculate PIQA, WinoGrande...")
        task_baseline(model, tokenizer)
    
if __name__ == "__main__":
    eval()

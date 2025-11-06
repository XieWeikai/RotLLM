import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from transformers import Trainer, default_data_collator
import datetime
import torch.distributed as dist
import transformers

from .optimizer import SGDG
from utils.data_utils import CustomJsonDataset 
from .prepare_model import prepare_model
from utils.process_args import process_args_ptq
from utils.utils import get_local_rank, log
from utils.adapt_mix_precision import collect_fakequant_configs


def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args, quant_configs = process_args_ptq()
    transformers.set_seed(ptq_args.seed)
    local_rank = get_local_rank()

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()

    device = "cuda"
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16

    model_orig = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_args.input_model, torch_dtype=dtype).to(device=device)

    tokenizer_classes = {
        "llama": "LlamaTokenizerFast",
        "qwen2": "Qwen2TokenizerFast",
    }
    tokenizer = None
    tokenizer_class_name = tokenizer_classes.get(model_orig.config.model_type)
    
    if tokenizer_class_name is not None:
        try:
            tokenizer_class = getattr(__import__('transformers'), tokenizer_class_name)
            if local_rank == 0:
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
            if local_rank == 0:
                log.info(f"✅ Successfully loaded {tokenizer_class.__name__}.")
        except Exception as e:
            if local_rank == 0:
                log.warning(f"Failed to load {tokenizer_class_name}: {e}")
            tokenizer = None
    
    # 如果加载 Fast tokenizer 失败，则回退到 AutoTokenizer
    if tokenizer is None:
        if local_rank == 0:
            log.info("✅ Using AutoTokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            add_eos_token=False,
            add_bos_token=False,
        )
        
    if local_rank == 0:
        log.info(f"Complete tokenizer loading...")

    # Prepare training data and calibration set.
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train_data = CustomJsonDataset(
        dataset["train"],
        tokenizer,
        block_size=min(training_args.model_max_length, 2048),
    )

    batch = None
    if ptq_args.mode == "static":
        # Prepare the calibration set for static quantization, used to initialize scale and zero_point
        num_samples = ptq_args.need_sample_for_static_init
        if ptq_args.adaptive_mixed_precision:
            num_samples += ptq_args.adapt_need_sample
        samples = [train_data[i + 10]["input_ids"] for i in range(num_samples)]
        batch = torch.tensor(samples).to(device=model_orig.device)

    # Prepare the trainable model and set parameters for training.
    model, R_trainable_parameters, q_trainable_parameters = prepare_model(
        model_orig,  
        quant_configs, 
        ptq_args,
        batch
    )
    model.train()

    if local_rank == 0:
        log.info("Model init completed for training...")
        log.info("💡Start to train...")
    
    # Applicable to RotLLM
    optimizer = SGDG(
        [
            {"params": R_trainable_parameters, "lr": training_args.learning_rate, "stiefel": True},
            {"params": q_trainable_parameters, "lr": training_args.learning_rate / 10}
        ],
        lr=training_args.learning_rate
    )

    MyTrainer = Trainer

    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=None,
        data_collator=default_data_collator,
        optimizers=(optimizer, None),
    )

    torch.distributed.barrier()

    trainer.train()

    cpu_state = trainer.model.state_dict()

    R_dict = {}
    for key, value in cpu_state.items():
        if "embed_tokens.R_post.weight" in key or "self_attn.v_proj.R_post.weight" in key:
            R_dict[key.replace(".weight", "")] =  value.clone().cpu()
        if "scale" in key or "zero_point" in key:
            R_dict[key] =  value.clone().cpu()
        
    fq_dict = collect_fakequant_configs(model, "txt/after_train_quant_config.txt", write_to_file=True)
    for key, value in fq_dict.items():
        R_dict[f"{key}.config.num_bits"] = value.num_bits

    if local_rank == 0:
        path = model_args.output_rotation_path
        dir_name = os.path.dirname(path)  
        os.makedirs(dir_name, exist_ok=True)  
        torch.save(
            R_dict,
            path,
        )

    dist.barrier()
    
if __name__ == "__main__":
    train()

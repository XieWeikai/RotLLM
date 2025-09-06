import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3, 4, 5"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from datasets import load_dataset
from transformers import Trainer, default_data_collator
import datetime
import torch.distributed as dist
from logging import Logger

from .optimizer import SGDG
from utils.data_utils import CustomJsonDataset 
from .prepare_model import prepare_model
from utils.process_args import process_args_ptq
from utils.utils import get_logger, get_local_rank

log: Logger = get_logger("RotLLM")

def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args, quant_configs = process_args_ptq()
    local_rank = get_local_rank()

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()

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
    if local_rank == 0:
        log.info(f"Complete tokenizer loading(GPU {local_rank})...")

    model_orig = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_args.input_model, torch_dtype=dtype).to(device=device)

    # Prepare training data and calibration set.
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train_data = CustomJsonDataset(
        dataset["train"],
        tokenizer,
        block_size=min(training_args.model_max_length, 2048),
    )

    # Prepare the trainable model and set parameters for training.
    model, R_trainable_parameters, q_trainable_parameters = prepare_model(model_orig, torch.tensor(train_data[0]["input_ids"]).unsqueeze(0).to(device=model_orig.device), quant_configs, ptq_args)
    model.train()

    if local_rank == 0:
        log.info("Model init completed for training...")
        log.info("Start to train...")
    
    # Applicable to RotLLM
    # optimizer = SGDG(
    #     [
    #         {"params": R_trainable_parameters, "lr": learning_rate, "stiefel": True},
    #         {"params": q_trainable_parameters, "lr": learning_rate}
    #     ],
    #     lr=learning_rate
    # )

    # Applicable to SpinQuant
    optimizer = SGDG(R_trainable_parameters, lr=training_args.learning_rate, stiefel=True)

    MyTrainer = Trainer

    # TODO: Adam(RotLLM)
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

    R_dict = {
        key.replace(".weight", ""): value.clone().cpu()
        for key, value in cpu_state.items()
        if "embed_tokens.R_post.weight" in key or "self_attn.v_proj.R_post.weight" in key
    }
    if local_rank == 0:
        os.makedirs(model_args.output_rotation_path, exist_ok=True)
        path = os.path.join(model_args.output_rotation_path, "RRR.bin")
        torch.save(
            R_dict,
            path,
        )

    dist.barrier()
    
if __name__ == "__main__":
    train()

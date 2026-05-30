import os
import torch
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoConfig, AutoProcessor
from datasets import load_dataset
from transformers import Trainer, default_data_collator
import datetime
import torch.distributed as dist
import transformers

from train.optimizer_sgd import SGDG
from utils.data_utils import CustomJsonDataset, get_wikitext2
from .prepare_model import prepare_model
from utils.process_args import process_args_ptq
from utils.utils import get_local_rank, log
from utils.adapt_mix_precision import collect_fakequant_configs
from evaluator.utils.evaluator import evaluator
from evaluator.task import task_baseline


def runner() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args, quant_configs = process_args_ptq()
    transformers.set_seed(ptq_args.seed)
    local_rank = get_local_rank()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()

    device = "cuda"
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32

    config = AutoConfig.from_pretrained(model_args.input_model, trust_remote_code=True)
    
    if local_rank == 0:
        log.info(f"📸 Loading Vision-Language Model (Type: {config.model_type})...")

    model_orig = AutoModelForVision2Seq.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model, 
        torch_dtype=dtype, 
        trust_remote_code=True
    ).to(device=device)

    # ================= 补丁：精选 text_config 核心属性提权到最外层 =================
    if hasattr(model_orig.config, "text_config"):
        if local_rank == 0:
            log.info("🛠️ Monkey-patching: Copying specific text_config attributes to main config...")
        
        # 纯文本量化和评估脚本最常强行读取的几个核心结构参数
        keys_to_patch = [
            "num_hidden_layers",     # 层数（必选）
            "hidden_size",           # 隐藏层维度
            "num_attention_heads",   # Q头数
            "num_key_value_heads",   # KV头数
            "intermediate_size",     # MLP中间层维度
        ]
        
        for key in keys_to_patch:    
            if not hasattr(model_orig.config, key):
                value = getattr(model_orig.config.text_config, key)
                setattr(model_orig.config, key, value)
            else:
                log.warning(f"🚨 Configuration conflict! Attribute '{key}' already exists in the main config and conflicts with a key in text_config.")
    # ======================================================================

    if local_rank == 0:
        log.info("✅ Using AutoTokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        add_eos_token=False,
        add_bos_token=False,
        trust_remote_code=True
    )
    if local_rank == 0:
        log.info(f"Complete tokenizer loading...")

    processor = AutoProcessor.from_pretrained(
        model_args.input_model,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True
    )
    if local_rank == 0:
        log.info(f"Complete processor loading...")

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
        if ptq_args.adaptive_mixed_precision or ptq_args.adaptive_online_rotation_R4 or ptq_args.adaptive_down_input_activation_16bits:
            num_samples += ptq_args.adapt_need_sample
        samples = [train_data[i + 10]["input_ids"] for i in range(num_samples)]
        batch = torch.tensor(samples).to(device=model_orig.device)

    # Prepare the trainable model and set parameters for training.
    model, adaptive_R4, R_trainable_parameters, q_trainable_parameters = prepare_model(
        model_orig,  
        dataset,
        quant_configs, 
        ptq_args,
        model_args,
        batch,
    )
    # model = model_orig
    # adaptive_R4, R_trainable_parameters, q_trainable_parameters = [], [], []

    # from infer.prompt_output import infer
    # infer(model, tokenizer)
    # assert False, "haha"

    # from infer.chat import infer
    # infer(model, tokenizer, "Could you please introduce the large language model?")
    # assert False, "haha"

    check_dict = collect_fakequant_configs(model, "txt/check_config.txt", write_to_file=True)

    if ptq_args.stage == "train":
        model.train()

        if local_rank == 0:
            log.info("Model init completed for training...")
            log.info("💡Start to train...")
        
        # for p in q_trainable_parameters:
        #     p.requires_grad_(False)

        # Applicable to RotLLM
        optimizer = SGDG(
            [
                {"params": R_trainable_parameters, "lr": training_args.learning_rate, "momentum": 0.0, "stiefel": True},
                {"params": q_trainable_parameters, "lr": training_args.learning_rate / 10, "momentum": 0.0, "nesterov": False},
            ],
            lr=training_args.learning_rate
        )
        # optimizer.param_names = {id(p): name for name, p in model.named_parameters()}
        # optimizer.local_rank = local_rank


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
            if "outActQuant" not in key and "qQuant" not in key and "kQuant" not in key and "vQuant" not in key:
                R_dict[f"{key}.config.num_bits"] = value.num_bits

        # Save the layers that require online rotation R4.
        R_dict["adaptive_R4"] = adaptive_R4       

        if local_rank == 0:
            path = model_args.output_rotation_path
            dir_name = os.path.dirname(path)  
            os.makedirs(dir_name, exist_ok=True)  
            torch.save(
                R_dict,
                path,
            )

        # if local_rank == 0:
        #     dict = {}
        #     from train.train_parameter import FakeQuantizer
        #     for name, module in model.named_modules():
        #         if isinstance(module, FakeQuantizer):
        #             if hasattr(module, "scale"):
        #                 dict[f"{name}.scale"] = module.scale
        #             if hasattr(module, "zero_point"):
        #                 dict[f"{name}.zero_point"] = module.zero_point
            
        #     with open("./txt/scale_train.txt", "w") as f:
        #         for k, v in dict.items():
        #             if v is None:
        #                 f.write(f"{k}: None\n")
        #             else:
        #                 # 标量 scale / zero_point
        #                 f.write(f"{k}: {v.detach().cpu().item()}\n")
    else:
        from evaluator.chat import chat
        chat(model, processor)

        log.info("Model init completed for evaling...")
        log.info("💡Start to eval...")
        
        if not ptq_args.task:
            dataset = load_dataset(
                "allenai/c4",
                "en",
                split="validation",
                streaming=True,
                trust_remote_code=True
            )

            dataset = dataset.take(800)
            from utils.data_utils import get_c4
            testloader = get_c4(
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
    
    dist.barrier()
    
if __name__ == "__main__":
    runner()
import os
import logging
from typing import Optional
import torch
from logging import Logger

# Define a utility method for setting the logging parameters of a logger
def get_logger(logger_name: Optional[str]) -> logging.Logger:
    # Get the logger with the specified name
    logger = logging.getLogger(logger_name)

    # Set the logging level of the logger to INFO
    logger.setLevel(logging.INFO)

    # Define a formatter for the log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a console handler for outputting log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger



def get_local_rank() -> int:
    if os.environ.get("LOCAL_RANK"):
        return int(os.environ["LOCAL_RANK"])
    else:
        logging.warning(
            "LOCAL_RANK from os.environ is None, fall back to get rank from torch distributed"
        )
        return torch.distributed.get_rank()


def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect

    caller_name = ""
    try:
        caller_name = f" (from {inspect.stack()[1].function})"
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(
            torch.cuda.memory_reserved(device=i)
            for i in range(torch.cuda.device_count())
        )

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )


def get_text_tower(model):
    """
    动态获取多模态模型或普通 LLM 中的文本塔主体模块及其路径。
    
    Args:
        model: Hugging Face 模型实例
        
    Returns:
        parent_path (str): 文本塔所在的字符串路径 (例如 "model.language_model")
        text_model (torch.nn.Module): 文本塔主体模块 (例如 model.model 或 model.language_model)
    """
    in_embed = model.get_input_embeddings()
    embed_name = next(name for name, mod in model.named_modules() if mod is in_embed)
    
    parent_path = embed_name.rsplit('.', 1)[0]
    text_model = model.get_submodule(parent_path)
    
    return parent_path, text_model


def set_config_attribute(model, attr_name: str, target_value) -> tuple:
    """
    修改模型的配置属性（同时兼容主 config 和 text_config），并返回修改前的原始状态。
    
    Args:
        model: 模型实例
        attr_name (str): 需要修改的属性名，例如 "use_cache"
        target_value: 想要设置的目标值，例如 False
        
    Returns:
        原始值，用于后续恢复。
    """
    original_states_main = None
    original_states_text = None


    # 1. 尝试从主 config 读取
    if hasattr(model.config, attr_name):
        original_states_main = getattr(model.config, attr_name, None)
        
    # 2. 尝试从 text_config 读取
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, attr_name):
        original_states_text = getattr(model.config.text_config, attr_name, None)

    # 3. 检查内外配置的值是否一致（仅当两者都存在时）
    if original_states_main is not None and original_states_text is not None and original_states_main != original_states_text:
        logging.warning(
            f"⚠️ 配置不一致警告: 属性 '{attr_name}' 在主 config 中的值为 {original_states_main}, "
            f"但在 text_config 中的值为 {original_states_text}。"
        )

    # 4. 执行修改（只修改本来就存在的属性，防止硬塞导致报错）
    if original_states_main is not None:
        setattr(model.config, attr_name, target_value)
    if original_states_text is not None:
        setattr(model.config.text_config, attr_name, target_value)

    return (original_states_main, original_states_text)


def restore_config_attribute(model, attr_name: str, original_states: tuple):
    """
    根据 set_config_attribute 返回的元组，将模型配置恢复到原始状态。
    
    Args:
        model: 模型实例
        attr_name (str): 需要恢复的属性名，例如 "use_cache"
        original_states: 之前保存的原始状态字典
    """
    original_states_main = original_states[0]
    original_states_text = original_states[1]
    
    # 1. 恢复主 config
    if original_states_main is not None:
        setattr(model.config, attr_name, original_states_main)

    if original_states_text is not None:
        setattr(model.config.text_config, attr_name, original_states_text)


log: Logger = get_logger("RotLLM")
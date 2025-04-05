import torch
from peft import PeftModel
from typing import Any, Tuple

from atomgpt.inverse_models.loader import FastLanguageModel
from jarvis.db.jsonutils import loadjson


def load_base_model(model_name: str, config) -> Tuple[torch.nn.Module, Any]:
    """
    Load the base model either from the Hugging Face Hub or a local directory.
    Delegates to a specialized loader for clarity and to follow SOLID principles.
    
    Parameters
    ----------
    model_name : str
        The model identifier (HF Hub name or local path).
    config : omegaconf.DictConfig
        The Hydra configuration, which includes:
            - model_source: "hf" or "local"
            - max_seq_length, dtype, load_in_4bit, etc.

    Returns
    -------
    model : torch.nn.Module
        The loaded model, possibly in 4-bit.
    tokenizer : transformers.Tokenizer
        The corresponding tokenizer.
    """
    loader = _get_model_loader(config.model_source)
    return loader(model_name, config)


def _get_model_loader(source: str):
    """
    Simple factory that returns the appropriate model loading function
    based on 'source'.
    """
    if source == "hf":
        return _load_from_hf
    elif source == "local":
        return _load_from_local
    else:
        raise ValueError(
            f"Invalid model_source '{source}'. Must be 'hf' or 'local'."
        )


def _load_from_hf(model_name: str, config) -> Tuple[torch.nn.Module, Any]:
    """
    Loads a model/tokenizer from the Hugging Face Hub using the
    FastLanguageModel utility.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
        # In case you need to trust custom code or pass tokens:
        trust_remote_code=True,
        # use_auth_token=True/False or local_files_only=False, etc.
    )
    return model, tokenizer


def _load_from_local(model_path: str, config) -> Tuple[torch.nn.Module, Any]:
    """
    Loads a model/tokenizer from a local directory.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
        # Usually for local we might specify local_files_only=True, 
        # but it's optional if your local path is definitely correct:
        local_files_only=True,
        trust_remote_code=False,
    )
    return model, tokenizer


def prepare_peft_model(model: torch.nn.Module, config) -> torch.nn.Module:
    """
    Wrap the base model in LoRA/PEFT if not already a PeftModel.

    Extensible for different PEFT strategies or configs.
    """
    if not isinstance(model, PeftModel):
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
    return model


def finalize_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """
    Set model to 'inference mode' for faster generation.

    Additional inference-time strategies (quantization, pruning) 
    can be integrated here.
    """
    FastLanguageModel.for_inference(model)
    return model


import torch
from peft import PeftModel
from typing import Any, Tuple

from atomgpt.inverse_models.loader import FastLanguageModel
from jarvis.db.jsonutils import loadjson


def load_base_model(model_name: str, config) -> Tuple[torch.nn.Module, Any]:
    """
    Load base (or 4-bit) model via FastLanguageModel.

    This function can be extended to support various model loading techniques
    without modifying this existing code (e.g., new ways to load HF models).
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )
    return model, tokenizer


def prepare_peft_model(model: torch.nn.Module, config) -> torch.nn.Module:
    """
    Wrap the base model in LoRA/PEFT if not already a PeftModel.

    New PEFT strategies or new LoRA configurations can be added by
    abstracting or extending from here, preserving OCP.
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

    Additional inference-time strategies (quantization, pruning) can
    be integrated as new steps in extended methods.
    """
    FastLanguageModel.for_inference(model)
    return model

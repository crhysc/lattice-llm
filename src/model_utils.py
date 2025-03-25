# model_utils.py
import torch
from peft import PeftModel
from atomgpt.inverse_models.loader import FastLanguageModel
from jarvis.db.jsonutils import loadjson
from typing import Optional

def load_base_model(model_name: str, config) -> "tuple[torch.nn.Module, Any]":
    """Loads base (or 4-bit) model via FastLanguageModel."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )
    return model, tokenizer

def prepare_peft_model(model, config):
    """Wrap the base model in LoRA/PEFT if not already a PeftModel."""
    if not isinstance(model, PeftModel):
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=[
                "q_proj","k_proj","v_proj","o_proj",
                "gate_proj","up_proj","down_proj",
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

def finalize_for_inference(model):
    """Sets model to 'inference mode' for faster generation."""
    FastLanguageModel.for_inference(model)
    return model

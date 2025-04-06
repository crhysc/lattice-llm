import torch
from peft import PeftModel
from typing import Any, Tuple
from abc import ABC, abstractmethod

from atomgpt.inverse_models.loader import FastLanguageModel

########################################################################
# 1) Abstract base class
########################################################################
class BaseModelLoader(ABC):
    """Abstract class for model loaders following SOLID principles."""

    @abstractmethod
    def load(self) -> Tuple[torch.nn.Module, Any]:
        """
        Return a tuple of (model, tokenizer).
        """
        pass


########################################################################
# 2) Concrete loader for Hugging Face Hub
########################################################################
class HfModelLoader(BaseModelLoader):
    def __init__(self, model_config, general_config):
        """
        model_config: e.g., cfg.model
        general_config: the entire Hydra config (if needed)
        """
        self.model_config = model_config
        self.general_config = general_config

    def load(self) -> Tuple[torch.nn.Module, Any]:
        """
        Loads model/tokenizer from Hugging Face Hub using FastLanguageModel.
        """
        model_name = self.model_config.remote_name
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.model_config.max_seq_length,
            dtype=self.model_config.dtype,
            load_in_4bit=self.model_config.load_in_4bit,
            trust_remote_code=True,
        )
        return model, tokenizer


########################################################################
# 3) Concrete loader for Local paths
########################################################################
class LocalModelLoader(BaseModelLoader):
    def __init__(self, model_config, general_config, override_path: str = None):
        """
        If override_path is provided, it can override the local path
        (e.g., if you're reloading a fine-tuned model from a saved dir).
        """
        self.model_config = model_config
        self.general_config = general_config
        self.override_path = override_path

    def load(self) -> Tuple[torch.nn.Module, Any]:
        """
        Loads model/tokenizer from a local directory.
        """
        model_path = self.override_path if self.override_path else self.model_config.local_name
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=self.model_config.max_seq_length,
            dtype=self.model_config.dtype,
            load_in_4bit=self.model_config.load_in_4bit,
            local_files_only=True,
            trust_remote_code=False,
        )
        return model, tokenizer


########################################################################
# 4) Loader Factory
########################################################################
def create_loader(model_config, general_config, load_from: str = None) -> BaseModelLoader:
    """
    Returns the appropriate loader instance for the specified model_source.
    `load_from` can be used to override local_name (e.g. reloading from 
    a saved fine-tuned model).
    """
    source = general_config.model_source

    if source == "hf":
        return HfModelLoader(model_config, general_config)
    elif source == "local":
        return LocalModelLoader(model_config, general_config, override_path=load_from)
    else:
        raise ValueError(
            f"Invalid model_source '{source}'. Must be 'hf' or 'local'."
        )

########################################################################
# 5) Primary load functions used by main
########################################################################
def load_base_model(model_config, general_config, load_from: str = None) -> Tuple[torch.nn.Module, Any]:
    """
    Create the appropriate loader, then load the model/tokenizer.
    """
    loader = create_loader(model_config, general_config, load_from)
    return loader.load()


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
    Switch model to inference mode for faster generation.
    Additional inference-time strategies (quantization, pruning) 
    can be integrated here.
    """
    FastLanguageModel.for_inference(model)
    return model


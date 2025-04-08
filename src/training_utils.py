# training_utils.py

from abc import ABC, abstractmethod
import torch
from transformers import TrainingArguments
from trl import SFTTrainer as HFSFTTrainer

from atomgpt.inverse_models.custom_trainer import CustomSFTTrainer


class AbstractSFTTrainer(ABC):
    """
    Abstract base class for any SFT trainer implementation.
    """

    @abstractmethod
    def train(self):
        """
        Run the training loop.
        """
        pass

    @abstractmethod
    def save_model(self, output_dir: str):
        """
        Save the trained model to the given output directory.
        """
        pass


class HuggingFaceSFTTrainer(AbstractSFTTrainer):
    """
    A thin wrapper around the original Hugging Face `SFTTrainer`,
    conforming to the AbstractSFTTrainer interface.
    """
    def __init__(self, model, tokenizer, dataset, config, logging_steps: int = 1):
        self.args = TrainingArguments(
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            overwrite_output_dir=True,
            learning_rate=config.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=logging_steps,
            optim=config.optim,
            weight_decay=0.01,
            lr_scheduler_type=config.lr_scheduler_type,
            seed=config.seed_val,
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            report_to="none",  # or tensorboard, etc.
        )

        self._trainer = HFSFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=config.model.max_seq_length,
            dataset_num_proc=config.dataset_num_proc,
            packing=False,
            args=self.args,
        )

    def train(self):
        """
        Invokes the underlying Hugging Face trainer's training loop.
        """
        return self._trainer.train()

    def save_model(self, output_dir: str):
        """
        Saves the underlying model to the specified output directory.
        """
        self._trainer.model.save_pretrained(output_dir)


class CustomWrappedSFTTrainer(AbstractSFTTrainer):
    """
    A thin wrapper around your `CustomSFTTrainer`, conforming to the
    AbstractSFTTrainer interface.
    """
    def __init__(self, model, tokenizer, dataset, config, logging_steps: int = 1):
        self.args = TrainingArguments(
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            overwrite_output_dir=True,
            learning_rate=config.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=logging_steps,
            optim=config.optim,
            weight_decay=0.01,
            lr_scheduler_type=config.lr_scheduler_type,
            seed=config.seed_val,
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            report_to="none",  # or tensorboard, etc.
        )

        self._trainer = CustomSFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=config.model.max_seq_length,
            dataset_num_proc=config.dataset_num_proc,
            loss_type=config.loss_type,
            packing=False,
            args=self.args,
        )

    def train(self):
        """
        Invokes the underlying custom trainer's training loop.
        """
        return self._trainer.train()

    def save_model(self, output_dir: str):
        """
        Saves the underlying model to the specified output directory.
        """
        self._trainer.model.save_pretrained(output_dir)


def create_sft_trainer(
    model,
    tokenizer,
    dataset,
    config,
    logging_steps: int = 1,
) -> AbstractSFTTrainer:
    """
    Factory function to create an appropriate SFT trainer depending
    on `config.trainer_type`.

    :param model: The loaded model
    :param tokenizer: The tokenizer corresponding to the model
    :param dataset: The training dataset
    :param config: Hydra/omegaconf config with trainer settings
    :param logging_steps: How often to log
    :return: An instance of AbstractSFTTrainer
    """
    if not hasattr(config, "trainer_type"):
        raise ValueError("`config.trainer_type` is not specified in the config.")

    trainer_type = config.trainer_type.lower()

    if trainer_type == "hf":
        return HuggingFaceSFTTrainer(model, tokenizer, dataset, config, logging_steps)
    elif trainer_type == "custom":
        return CustomWrappedSFTTrainer(model, tokenizer, dataset, config, logging_steps)
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")


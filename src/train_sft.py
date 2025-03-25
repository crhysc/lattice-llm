import torch
from trl import SFTTrainer
from transformers import TrainingArguments

from atomgpt.inverse_models.custom_trainer import CustomSFTTrainer


def create_sft_trainer(model,
                       tokenizer,
                       dataset,
                       config,
                       logging_steps: int = 1) -> CustomSFTTrainer:
    """
    Create the SFTTrainer with user-defined arguments.

    If you want multiple trainer configurations, you can create different
    factories that return trainers without editing existing logic here.
    """
    trainer = CustomSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        dataset_num_proc=config.dataset_num_proc,
        loss_type=config.loss_type,
        packing=False,  # Turn on if you want auto-packing
        args=TrainingArguments(
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=5,
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
            report_to="none",
        ),
    )
    return trainer

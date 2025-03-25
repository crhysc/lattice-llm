import os
import shutil
import tempfile
from typing import Any, Dict

import optuna
import numpy as np
from pydantic_settings import BaseSettings
from datasets import load_dataset
from transformers import TrainingArguments

# Local imports
from data_utils import (
    load_id_prop_data,
    make_alpaca_json,
    formatting_prompts_func
)
from model_utils import (
    load_base_model,
    prepare_peft_model,
    finalize_for_inference
)
from train_sft import create_sft_trainer
from evaluate import evaluate
from jarvis.db.jsonutils import dumpjson, loadjson


class TrainingPropConfig(BaseSettings):
    """
    Configuration fields for your training/hyperparameter search.

    Override some or all of these values (e.g., from JSON) for different runs.
    """

    # Paths, model names, etc.
    id_prop_path: str = "atomgpt/examples/inverse_model/id_prop.csv"
    model_name: str = "knc6/atomgpt_mistral_tc_supercon"
    output_dir: str = "outputs"
    model_save_path: str = "lora_model_m"
    csv_out: str = "AI-AtomGen-prop-dft_3d-test-rmse.csv"

    # Prompt format
    alpaca_prompt: str = "### Instruction:\n{}\n### Input:\n{}\n### Output:\n{}"
    instruction: str = "Below is a description of a superconductor material."
    output_prompt: str = " Generate atomic structure description..."

    # Data-related properties
    prop: str = "Tc_supercon"
    id_tag: str = "id"
    chem_info: str = "formula"   # "none", "formula", or "element_list"
    file_format: str = "poscar"
    num_train: int = 2
    num_test: int = 2

    # HPC / GPU settings
    max_seq_length: int = 2048
    dtype: str = None
    load_in_4bit: bool = True

    # Trainer defaults
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 2
    logging_steps: int = 10
    dataset_num_proc: int = 2
    loss_type: str = "default"
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "linear"
    seed_val: int = 3407

    # LoRA / PEFT config
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0


def hyperparam_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Define the range of hyperparameters Optuna will search over.
    Extend or modify this function to explore different parameters.
    """
    return {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 5e-4),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [1, 2, 4]
        ),
        "gradient_accumulation_steps": trial.suggest_categorical(
            "gradient_accumulation_steps", [1, 2, 4, 8]
        ),
        "lora_rank": trial.suggest_categorical("lora_rank", [8, 16, 32]),
        "num_epochs": trial.suggest_int("num_epochs", 1, 4),
    }


def objective(trial: optuna.Trial, base_config: TrainingPropConfig) -> float:
    """
    The function that Optuna repeatedly calls, each time with new hyperparams.

    Returns a metric to be minimized (e.g. validation loss).
    """
    # Copy the base config so we don't globally modify it
    cfg = base_config.copy()

    # Sample hyperparameters from the search space
    sampled_params = hyperparam_search_space(trial)
    cfg.learning_rate = sampled_params["learning_rate"]
    cfg.per_device_train_batch_size = sampled_params["per_device_train_batch_size"]
    cfg.gradient_accumulation_steps = sampled_params["gradient_accumulation_steps"]
    cfg.lora_rank = sampled_params["lora_rank"]
    cfg.num_epochs = sampled_params["num_epochs"]

    # Create a temporary directory for each trial
    temp_dir = tempfile.mkdtemp(prefix="optuna_trial_")
    cfg.output_dir = os.path.join(temp_dir, "output")
    cfg.model_save_path = os.path.join(temp_dir, "model_save")
    cfg.csv_out = os.path.join(temp_dir, "eval_results.csv")
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.model_save_path, exist_ok=True)

    try:
        # ---------- DATA LOADING ----------
        data_list = load_id_prop_data(cfg.id_prop_path, cfg)
        train_ids = [row["id"] for row in data_list[: cfg.num_train]]
        test_ids = [row["id"] for row in data_list[
            cfg.num_train : cfg.num_train + cfg.num_test
        ]]

        # Create train set in Alpaca JSON format
        alpaca_prop_train = os.path.join(cfg.output_dir, "alpaca_prop_train.json")
        if not os.path.exists(alpaca_prop_train):
            m_train = make_alpaca_json(data_list, train_ids, cfg)
            dumpjson(m_train, alpaca_prop_train)
        else:
            m_train = loadjson(alpaca_prop_train)

        # Create test set
        alpaca_prop_test = os.path.join(cfg.output_dir, "alpaca_prop_test.json")
        if not os.path.exists(alpaca_prop_test):
            m_test = make_alpaca_json(data_list, test_ids, cfg)
            dumpjson(m_test, alpaca_prop_test)
        else:
            m_test = loadjson(alpaca_prop_test)

        # ---------- MODEL LOADING ----------
        model, tokenizer = load_base_model(cfg.model_name, cfg)
        model = prepare_peft_model(model, cfg)

        # ---------- DATASET PREP ----------
        dataset = load_dataset("json", data_files=alpaca_prop_train, split="train")
        dataset = dataset.map(
            lambda examples: formatting_prompts_func(examples, cfg.alpaca_prompt),
            batched=True,
        )

        # ---------- TRAINING ----------
        trainer = create_sft_trainer(model, tokenizer, dataset, cfg)
        trainer.train()
        model.save_pretrained(cfg.model_save_path)

        # ---------- EVALUATION ----------
        # Reload for inference
        model, tokenizer = load_base_model(cfg.model_save_path, cfg)
        model = finalize_for_inference(model)

        # Evaluate on test set
        evaluate(m_test, model, tokenizer, cfg.csv_out, cfg)

        # For demonstration, we return a random value as the “metric”
        # In practice, parse your CSV and compute a real error metric.
        val_loss = float(np.random.rand())
        return val_loss

    finally:
        # Cleanup temp directory after trial
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_hyperparameter_search(config_path: str, n_trials: int = 5) -> optuna.Study:
    """
    Load a base config, set up an Optuna study, and run multiple trials.

    :param config_path: Path to a JSON config file.
    :param n_trials: Number of trials for Optuna to run.
    """
    base_cfg_dict = loadjson(config_path)
    base_cfg = TrainingPropConfig(**base_cfg_dict)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, base_cfg), n_trials=n_trials)

    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")
    return study


if __name__ == "__main__":
    # Example usage
    run_hyperparameter_search(
        config_path="alignn/examples/sample_data/config_example.json",
        n_trials=3
    )

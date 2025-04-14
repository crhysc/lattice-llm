# hyperparameter_search.py

import os
import time
import shutil
import tempfile
import numpy as np
import optuna

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from transformers import IntervalStrategy, TrainingArguments
from datasets import load_dataset

from resolvers import register_resolvers
from jarvis.db.jsonutils import dumpjson, loadjson

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
from training_utils import create_sft_trainer
from evaluate import evaluate


# ---------------------------------------------------------------------
# 1) Convergence Metric Strategies
# ---------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import List


class BaseConvergenceMetric(ABC):
    """Abstract base class to compute a single scalar from a list of val losses."""

    @abstractmethod
    def compute_metric(self, val_losses: List[float]) -> float:
        """
        Given the per-epoch validation losses, return a single scalar
        to measure convergence performance.
        """
        pass


class FinalValLossMetric(BaseConvergenceMetric):
    """Use the final epoch's validation loss."""

    def compute_metric(self, val_losses: List[float]) -> float:
        if not val_losses:
            return float("inf")  # if somehow no val losses are tracked
        return val_losses[-1]


class AreaUnderCurveMetric(BaseConvergenceMetric):
    """Use the sum of validation losses over epochs (simple AUC)."""

    def compute_metric(self, val_losses: List[float]) -> float:
        if not val_losses:
            return float("inf")
        # Optionally normalize by number of epochs
        return sum(val_losses)


class SlopeValLossMetric(BaseConvergenceMetric):
    """
    Fit a linear slope to the validation losses across epochs.
    A more negative slope can indicate continuing convergence.
    We'll return the absolute value of the slope so that smaller is better
    if you are 'minimizing' (or you can invert it to taste).
    """

    def compute_metric(self, val_losses: List[float]) -> float:
        if len(val_losses) < 2:
            return float("inf")
        import numpy as np

        x = np.arange(len(val_losses))
        # Fit a line: val_loss ~ slope*x + intercept
        slope, _ = np.polyfit(x, val_losses, 1)
        # If you want to favor more negative slope => smaller is better, use abs
        return abs(slope)


class ConvergenceMetricFactory:
    """Creates the appropriate convergence metric strategy from a string."""

    @staticmethod
    def create(metric_name: str) -> BaseConvergenceMetric:
        metric_name = metric_name.lower()
        if metric_name == "final_val_loss":
            return FinalValLossMetric()
        elif metric_name == "auc":
            return AreaUnderCurveMetric()
        elif metric_name == "slope":
            return SlopeValLossMetric()
        else:
            raise ValueError(f"Unknown convergence metric: {metric_name}")


# ---------------------------------------------------------------------
# 2) Optuna Objective / Hyperparameter Search
# ---------------------------------------------------------------------
def hyperparam_search_space(trial: optuna.Trial, cfg: DictConfig):
    """
    Define how Optuna suggests hyperparameters.
    You can expand or modify this to handle additional parameters
    from your Hydra config if needed.
    """
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 5e-4)
    batch_size = trial.suggest_categorical("per_device_train_batch_size", [1, 2, 4])
    grad_accum = trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4, 8])
    lora_rank = trial.suggest_categorical("lora_rank", [8, 16, 32])
    num_epochs = trial.suggest_int("num_epochs", 1, 4)

    return {
        "learning_rate": lr,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "lora_rank": lora_rank,
        "num_epochs": num_epochs,
    }


def objective(trial: optuna.Trial, cfg: DictConfig) -> float:
    """
    The objective function that Optuna calls each trial with new hyperparams.
    Returns the scalar metric to be minimized (or maximized, per config).
    """
    # Copy the user config so we don’t pollute the global Hydra config
    local_cfg = OmegaConf.to_container(cfg, resolve=True)
    local_cfg = OmegaConf.create(local_cfg)

    # Sample hyperparams from the search space
    sampled_params = hyperparam_search_space(trial, local_cfg)
    local_cfg.learning_rate = sampled_params["learning_rate"]
    local_cfg.per_device_train_batch_size = sampled_params["per_device_train_batch_size"]
    local_cfg.gradient_accumulation_steps = sampled_params["gradient_accumulation_steps"]
    local_cfg.lora_rank = sampled_params["lora_rank"]
    local_cfg.num_epochs = sampled_params["num_epochs"]

    # Create a temporary directory for outputs in this trial
    temp_dir = tempfile.mkdtemp(prefix="optuna_trial_")
    local_cfg.output_dir = os.path.join(temp_dir, "output")
    local_cfg.model_save_path = os.path.join(temp_dir, "model_save")
    local_cfg.csv_out = os.path.join(temp_dir, "eval_results.csv")
    os.makedirs(local_cfg.output_dir, exist_ok=True)
    os.makedirs(local_cfg.model_save_path, exist_ok=True)

    # Pick the user's chosen convergence metric
    metric_strategy = ConvergenceMetricFactory.create(
        local_cfg.get("convergence_metric", "final_val_loss")
    )

    try:
        # ------------------- Data Loading -------------------
        data_list = load_id_prop_data(local_cfg.id_prop_path, local_cfg)

        # Example: simple splitting using num_train/num_test from config
        train_ids = [
            row["id"] for row in data_list[: local_cfg.num_train]
        ]
        test_ids = [
            row["id"] for row in data_list[
                local_cfg.num_train : local_cfg.num_train + local_cfg.num_test
            ]
        ]

        # Create train set (Alpaca JSON)
        alpaca_prop_train = os.path.join(local_cfg.output_dir, "alpaca_prop_train.json")
        if not os.path.exists(alpaca_prop_train):
            train_data = make_alpaca_json(data_list, train_ids, local_cfg)
            dumpjson(train_data, alpaca_prop_train)
        else:
            train_data = loadjson(alpaca_prop_train)

        # Create test set
        alpaca_prop_test = os.path.join(local_cfg.output_dir, "alpaca_prop_test.json")
        if not os.path.exists(alpaca_prop_test):
            test_data = make_alpaca_json(data_list, test_ids, local_cfg)
            dumpjson(test_data, alpaca_prop_test)
        else:
            test_data = loadjson(alpaca_prop_test)

        # ------------------- Load Model -------------------
        model, tokenizer = load_base_model(local_cfg.model, local_cfg)
        model = prepare_peft_model(model, local_cfg)

        # ------------------- Prepare Training Dataset -------------------
        dataset = load_dataset("json", data_files=alpaca_prop_train, split="train")
        dataset = dataset.map(
            lambda examples: formatting_prompts_func(examples, local_cfg.alpaca_prompt),
            batched=True,
        )

        # ------------------- Custom Trainer Args to Evaluate Each Epoch -------------------
        # We can do this by customizing the default arguments if needed.
        # Then create our SFT trainer with these arguments:
        hf_training_args = TrainingArguments(
            output_dir=local_cfg.output_dir,
            num_train_epochs=local_cfg.num_epochs,
            per_device_train_batch_size=local_cfg.per_device_train_batch_size,
            gradient_accumulation_steps=local_cfg.gradient_accumulation_steps,
            evaluation_strategy=IntervalStrategy.EPOCH,  # Evaluate each epoch
            save_strategy=IntervalStrategy.NO,           # Not saving per epoch
            logging_steps=10,
            learning_rate=local_cfg.learning_rate,
            weight_decay=0.01,
            warmup_ratio=local_cfg.warmup_ratio,
            lr_scheduler_type=local_cfg.lr_scheduler_type,
            optim=local_cfg.optim,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            report_to="none",  # or "tensorboard"
            seed=local_cfg.seed_val,
        )

        # We'll create a trainer but override the arguments
        trainer = create_sft_trainer(
            model,
            tokenizer,
            dataset,
            config=local_cfg,
            logging_steps=hf_training_args.logging_steps,
        )
        # Overwrite the underlying trainer's arguments
        trainer._trainer.args = hf_training_args

        # Now train - the trainer should run validation after each epoch,
        # and log_history will store the "eval_loss".
        trainer.train()

        # We can collect the validation losses from the trainer logs:
        val_losses = []
        for entry in trainer._trainer.state.log_history:
            if "eval_loss" in entry:
                val_losses.append(entry["eval_loss"])

        # Save the final LoRA model
        model.save_pretrained(local_cfg.model_save_path)

        # --------------- Compute the Convergence Metric ---------------
        # e.g. final_val_loss, sum(AUC), slope, etc.
        val_metric = metric_strategy.compute_metric(val_losses)

        # --------------- Evaluate the final model for other tasks ---------------
        # This is separate from the convergence metric:
        model, tokenizer = load_base_model(local_cfg.model_save_path, local_cfg)
        model = finalize_for_inference(model)
        evaluate(test_data, model, tokenizer, local_cfg.csv_out, local_cfg)

        # Return the metric that Optuna will minimize or maximize
        return val_metric

    finally:
        # Clean up the temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@hydra.main(version_base=None, config_path="../configs", config_name="hyperparam_search")
def main(cfg: DictConfig):
    """
    Launch an Optuna hyperparameter search using Hydra for config.
    The user can pick among "final_val_loss", "auc", or "slope"
    via cfg.convergence_metric (default: final_val_loss).
    """
    register_resolvers()

    print("***** Hyperparameter Search Configuration *****")
    print(OmegaConf.to_yaml(cfg))

    # Ensure a GPU is available
    if not torch.cuda.is_available():
        raise ValueError("GPU not available. This script is intended for GPU training.")

    start_time = time.time()

    # Create Optuna study: direction should be 'minimize' for loss
    # or 'maximize' if your metric is an accuracy-like measure.
    direction = cfg.get("study_direction", "minimize")
    study = optuna.create_study(direction=direction)

    # Number of trials from config or default
    n_trials = cfg.get("n_trials", 5)
    study.optimize(lambda t: objective(t, cfg), n_trials=n_trials)

    # Print the best result
    print(f"Best trial value: {study.best_value}")
    print(f"Best trial params: {study.best_params}")

    elapsed = time.time() - start_time
    print(f"Done! Total time: {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()

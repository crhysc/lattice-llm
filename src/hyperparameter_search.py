# hyperparameter_search.py
import optuna
import os
from model_utils import load_base_model, prepare_peft_model
from train_sft import create_sft_trainer
...

def hyperparam_search(trial):
    """Define search space."""
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 5e-4)
    batch_size = trial.suggest_categorical("batch_size", [2,4,8])
    ...
    return {"learning_rate": lr, "per_device_train_batch_size": batch_size, ...}

def run_hpo(config):
    def objective(trial):
        hparams = hyperparam_search(trial)
        # Modify config in-place or create a new config
        config.learning_rate = hparams["learning_rate"]
        config.per_device_train_batch_size = hparams["per_device_train_batch_size"]
        # Load model
        model, tokenizer = load_base_model(config.model_name, config)
        model = prepare_peft_model(model, config)
        # Prepare data ...
        trainer = create_sft_trainer(model, tokenizer, dataset, config)
        result = trainer.train()
        metrics = trainer.evaluate()  # If you have a separate validation set
        return metrics["eval_loss"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    print("Best params:", study.best_params)
    return study.best_params

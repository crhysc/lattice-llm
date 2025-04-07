import os
import time
import torch

import hydra
from omegaconf import DictConfig, OmegaConf
from jarvis.db.jsonutils import dumpjson, loadjson
from datasets import load_dataset

# Local imports
from data_utils import (
    make_alpaca_json,
    formatting_prompts_func,
    load_id_prop_data,
)
from model_utils import (
    load_base_model,
    prepare_peft_model,
    finalize_for_inference,
)
from resolvers import register_resolvers
from train_sft import create_sft_trainer
from evaluate import evaluate


config_path = os.path.join(os.path.dirname(__file__), "../configs")
register_resolvers()


@hydra.main(version_base=None, config_path=config_path, config_name="main")
def main(cfg: DictConfig):
    """
    Main training/evaluation pipeline.

    Each step can be extended with new logic (e.g., alternative training sets,
    alternative model load functions) by introducing new modules/factories
    rather than changing this function's core flow.
    """
    if not torch.cuda.is_available():
        raise ValueError("GPU not available. Training only tested with GPU.")

    # Print the merged config (for debugging)
    print("***** Merged Configuration *****")
    print(OmegaConf.to_yaml(cfg))

    start_time = time.time()

    # 1) Make output directories
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.model_save_path, exist_ok=True)

    # 2) Save config for reproducibility
    dumpjson(
        OmegaConf.to_container(cfg, resolve=True),
        os.path.join(cfg.output_dir, "atomgpt_config.json"),
    )

    # 3) Load data
    data_list = load_id_prop_data(cfg.id_prop_path, cfg)

    # Example train/test splits
    train_ids = [row["id"] for row in data_list[: cfg.num_train]]
    test_ids = [
        row["id"] for row in data_list[cfg.num_train : cfg.num_train + cfg.num_test]
    ]

    # 4) Create Alpaca-style JSON for train
    alpaca_prop_train = os.path.join(cfg.output_dir, "alpaca_prop_train.json")
    if not os.path.exists(alpaca_prop_train):
        train_data = make_alpaca_json(data_list, train_ids, cfg)
        dumpjson(train_data, alpaca_prop_train)
    else:
        train_data = loadjson(alpaca_prop_train)

    # 5) Create Alpaca-style JSON for test
    alpaca_prop_test = os.path.join(cfg.output_dir, "alpaca_prop_test.json")
    if not os.path.exists(alpaca_prop_test):
        test_data = make_alpaca_json(data_list, test_ids, cfg)
        dumpjson(test_data, alpaca_prop_test)
    else:
        test_data = loadjson(alpaca_prop_test)

    # 6) Load model
    model, tokenizer = load_base_model(cfg.model, cfg)
    model = prepare_peft_model(model, cfg)

    # 7) Prepare training dataset
    dataset = load_dataset("json", data_files=alpaca_prop_train, split="train")
    dataset = dataset.map(
        lambda examples: formatting_prompts_func(examples, cfg.alpaca_prompt),
        batched=True,
    )

    # 8) Create trainer and train
    trainer = create_sft_trainer(model, tokenizer, dataset, cfg)
    trainer.train()
    model.save_pretrained(cfg.model_save_path)

    # 9) Evaluate - re-load in inference mode
    model, tokenizer = load_base_model(cfg.model_save_path, cfg)
    model = finalize_for_inference(model)
    evaluate(test_data, model, tokenizer, cfg.csv_out, cfg)

    print(f"Done! Total time: {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()

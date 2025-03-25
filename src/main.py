# src/main.py

import os
import time
import torch

import hydra
from omegaconf import DictConfig, OmegaConf

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
from train_sft import create_sft_trainer
from evaluate import evaluate
from datasets import load_dataset
from jarvis.db.jsonutils import dumpjson, loadjson


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):

    if not torch.cuda.is_available():
        raise ValueError("GPU not available. Training only tested with GPU.")

    # Print the merged config (for debugging)
    print("***** Merged Configuration *****")
    print(OmegaConf.to_yaml(cfg))

    t0 = time.time()

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
    test_ids  = [row["id"] for row in data_list[cfg.num_train : cfg.num_train + cfg.num_test]]

    # 4) Create Alpaca-style JSON for train
    alpaca_prop_train = os.path.join(cfg.output_dir, "alpaca_prop_train.json")
    if not os.path.exists(alpaca_prop_train):
        m_train = make_alpaca_json(data_list, train_ids, cfg)
        dumpjson(m_train, alpaca_prop_train)
    else:
        m_train = loadjson(alpaca_prop_train)

    # 5) Create Alpaca-style JSON for test
    alpaca_prop_test = os.path.join(cfg.output_dir, "alpaca_prop_test.json")
    if not os.path.exists(alpaca_prop_test):
        m_test = make_alpaca_json(data_list, test_ids, cfg)
        dumpjson(m_test, alpaca_prop_test)
    else:
        m_test = loadjson(alpaca_prop_test)

    # 6) Load model
    model, tokenizer = load_base_model(cfg.model_name, cfg)
    model = prepare_peft_model(model, cfg)

    # 7) Prepare training dataset
    dataset = load_dataset("json", data_files=alpaca_prop_train, split="train")
    dataset = dataset.map(
        lambda examples: formatting_prompts_func(examples, cfg.alpaca_prompt),
        batched=True,
    )

    # 8) Create trainer and train
    trainer = create_sft_trainer(model, tokenizer, dataset, cfg)
    trainer_stats = trainer.train()
    model.save_pretrained(cfg.model_save_path)

    # 9) Evaluate
    #    Re-load in inference mode
    model, tokenizer = load_base_model(cfg.model_save_path, cfg)
    model = finalize_for_inference(model)
    evaluate(m_test, model, tokenizer, cfg.csv_out, cfg)

    print(f"Done! Total time: {time.time() - t0:.2f} seconds.")


if __name__ == "__main__":
    main()

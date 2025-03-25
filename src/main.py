# main.py
import os
import json
import pprint
import argparse
import sys
import time

# Local imports
from data_utils import (make_alpaca_json, formatting_prompts_func,
                        load_id_prop_data)
from model_utils import (load_base_model, prepare_peft_model,
                         finalize_for_inference)
from train_sft import create_sft_trainer
from evaluate import evaluate

from datasets import load_dataset
from jarvis.db.jsonutils import loadjson, dumpjson
from pydantic_settings import BaseSettings


class TrainingPropConfig(BaseSettings):
    """Your same config fields."""
    # [ All the fields you have in your original script... ]
    # e.g.
    id_prop_path: str = "atomgpt/examples/inverse_model/id_prop.csv"
    # etc...


def parse_args():
    parser = argparse.ArgumentParser(description="Atomistic GPT Training.")
    parser.add_argument(
        "--config_name",
        default="alignn/examples/sample_data/config_example.json",
        help="Name of the config file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise ValueError("GPU not available. Training only tested with GPU.")

    print("Loading config:", args.config_name)
    raw_cfg = loadjson(args.config_name)
    config = TrainingPropConfig(**raw_cfg)
    pprint.pprint(config.dict())

    t0 = time.time()
    # Make output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.model_save_path, exist_ok=True)

    # Save config for reproducibility
    dumpjson(config.dict(), os.path.join(config.output_dir, "atomgpt_config.json"))
    dumpjson(config.dict(), os.path.join(config.model_save_path, "atomgpt_config.json"))

    # 1) Load data
    data_list = load_id_prop_data(config.id_prop_path, config)
    # Example split
    train_ids = [row["id"] for row in data_list[: config.num_train]]
    test_ids  = [row["id"] for row in data_list[config.num_train : config.num_train+config.num_test]]

    # 2) Create Alpaca-style JSON
    alpaca_prop_train = os.path.join(config.output_dir, "alpaca_prop_train.json")
    if not os.path.exists(alpaca_prop_train):
        m_train = make_alpaca_json(data_list, train_ids, config)
        dumpjson(m_train, alpaca_prop_train)
    else:
        m_train = loadjson(alpaca_prop_train)

    alpaca_prop_test = os.path.join(config.output_dir, "alpaca_prop_test.json")
    if not os.path.exists(alpaca_prop_test):
        m_test = make_alpaca_json(data_list, test_ids, config)
        dumpjson(m_test, alpaca_prop_test)
    else:
        m_test = loadjson(alpaca_prop_test)

    # 3) Load model
    model, tokenizer = load_base_model(config.model_name, config)
    model = prepare_peft_model(model, config)

    # 4) Setup dataset for training
    dataset = load_dataset("json", data_files=alpaca_prop_train, split="train")
    dataset = dataset.map(
        lambda examples: formatting_prompts_func(examples, config.alpaca_prompt),
        batched=True,
    )

    # 5) Create trainer and train
    trainer = create_sft_trainer(model, tokenizer, dataset, config)
    trainer_stats = trainer.train()
    model.save_pretrained(config.model_save_path)

    # 6) Evaluate
    model, tokenizer = load_base_model(config.model_save_path, config)
    model = finalize_for_inference(model)
    # Evaluate with single-sample or batch approach
    evaluate(m_test, model, tokenizer, config.csv_out, config)

    print(f"Done! Total time: {time.time() - t0:.2f} seconds.")


if __name__ == "__main__":
    main()

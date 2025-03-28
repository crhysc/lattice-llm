"""
main.py

Entry point that utilizes Hydra for configuration and orchestrates 
the data preprocessing workflow.
"""

import logging
import hydra
from omegaconf import DictConfig

from utils.data_preprocessor import process_superconducting_dataset

# Optional: you can configure logging more thoroughly
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
def main_app(cfg: DictConfig) -> None:
    """
    Main entry point, orchestrating the data preprocessing.
    Hydra automatically parses the config.yaml and 
    any command-line overrides into `cfg`.
    """
    input_dir = cfg.input_dir
    output_dir = cfg.output_dir
    filename_prefix = cfg.filename_prefix
    tc_key = cfg.tc_key

    process_superconducting_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        filename_prefix=filename_prefix,
        tc_key=tc_key,
    )


if __name__ == "__main__":
    main_app()


# slurm/utils.py

"""
utils.py
========

Utility functions for loading SLURM templates, filling placeholders,
generating job names, saving files, and basic file/path validation.
"""

import datetime
from pathlib import Path

def load_template(template_path: str) -> str:
    """
    Load the contents of a SLURM template file into a single string.
    
    :param template_path: Path to the .sbatch template (e.g. 'slurm/templates/train_template.sbatch')
    :return: A string containing the full text of the template.
    """
    return Path(template_path).read_text()

def fill_template(template: str, params: dict) -> str:
    """
    Fill placeholders in the template using Python's built-in format.

    :param template: The string template with placeholders like {job_name}, {script}, etc.
    :param params: A dictionary of placeholder->value mappings
    :return: The formatted template as a string
    """
    return template.format(**params)

def generate_job_name(base: str) -> str:
    """
    Generate a unique job name by appending a timestamp to the base name.
    
    :param base: The base string for the job name (e.g., 'lattice_train')
    :return: A string like 'lattice_train_20250323-104500'
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{base}_{timestamp}"

def save_script(script_text: str, output_path: Path) -> None:
    """
    Save the generated SLURM script to a file, creating directories if needed.
    
    :param script_text: The complete text to be written to the .sbatch file
    :param output_path: The Path object where the file should be written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(script_text)

def validate_file_exists(path: str) -> None:
    """
    Raise a FileNotFoundError if the given file does not exist.

    :param path: Path to the file to validate.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


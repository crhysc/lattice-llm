#!/usr/bin/env python3

"""
submit_job.py
=============

Command-line script to generate SLURM batch files from templates
and optionally submit them to a SLURM queue via sbatch.

Now enhanced to pull defaults from a YAML file and allow CLI overrides.
"""

import argparse
import subprocess
from pathlib import Path
import yaml  # <-- NEW: for loading defaults.yaml

# Import utilities from utils.py (assuming you have them)
from slurm.utils import (
    load_template,
    fill_template,
    generate_job_name,
    save_script,
    validate_file_exists
)

def main():
    # 1. Try loading defaults from defaults.yaml
    defaults_file = Path(__file__).parent / "defaults.yaml"
    if defaults_file.exists():
        with open(defaults_file, "r") as f:
            defaults = yaml.safe_load(f)
    else:
        # If you want, you can raise an error or just use an empty dict
        defaults = {}

    # 2. Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate and optionally submit a SLURM job."
    )

    # For each argument, we fetch the default from the YAML if it exists; otherwise fall back
    parser.add_argument("--job_name", 
                        default=defaults.get("job_name", None),
                        help="Base name for the job; a timestamp is appended automatically.")
    parser.add_argument("--template", 
                        default=defaults.get("template", "slurm/templates/train_template.sbatch"),
                        help="Path to the SLURM template .sbatch file.")
    parser.add_argument("--script", 
                        default=defaults.get("script", "src/main.py"),
                        help="Path to the script you want to run inside SLURM.")
    parser.add_argument("--config", 
                        default=defaults.get("config", "config.yaml"),
                        help="Path to a config file (if your script requires one).")

    # SLURM resource options
    parser.add_argument("--time", 
                        default=defaults.get("time", "01:00:00"),
                        help="Requested runtime for the job.")
    parser.add_argument("--partition", 
                        default=defaults.get("partition", "gpu"),
                        help="SLURM partition to submit to (e.g., 'gpu' or 'compute').")
    parser.add_argument("--gpus", 
                        default=defaults.get("gpus", "1"),
                        help="Number of GPUs requested (if using GPUs).")
    parser.add_argument("--mem", 
                        default=defaults.get("mem", "16G"),
                        help="Memory to allocate for the job.")
    parser.add_argument("--cpus", 
                        default=defaults.get("cpus", "4"),
                        help="Number of CPU cores per task.")

    # Output control
    parser.add_argument("--output", 
                        default=defaults.get("output", "slurm/generated/job.sbatch"),
                        help="Where to save the generated .sbatch file.")
    parser.add_argument("--submit", 
                        action="store_true",
                        help="If specified, automatically submit the job with 'sbatch'.")

    args = parser.parse_args()

    # 3. Validate that required files exist
    validate_file_exists(args.template)
    validate_file_exists(args.script)
    validate_file_exists(args.config)

    # 4. Load and fill the template
    template_text = load_template(args.template)
    job_name = generate_job_name(args.job_name)
    filled_text = fill_template(template_text, {
        "job_name": job_name,
        "script": args.script,
        "config": args.config,
        "time": args.time,
        "partition": args.partition,
        "gpus": args.gpus,
        "mem": args.mem,
        "cpus": args.cpus
    })

    # 5. Save the generated script
    output_path = Path(args.output)
    save_script(filled_text, output_path)

    # 6. Optionally submit
    if args.submit:
        print(f"Submitting {output_path} to SLURM queue...")
        subprocess.run(["sbatch", str(output_path)])
    else:
        print(f"SLURM script generated at: {output_path}\n"
              f"Use 'sbatch {output_path}' to submit manually.")

if __name__ == "__main__":
    main()


"""
submit_job.py
=============

Command-line script to generate SLURM batch files from templates
and optionally submit them to a SLURM queue via sbatch.
"""

import argparse
import subprocess
from pathlib import Path

# Import the utilities from utils.py
from slurm.utils import (
    load_template,
    fill_template,
    generate_job_name,
    save_script,
    validate_file_exists
)

def main():
    parser = argparse.ArgumentParser(
        description="Generate and optionally submit a SLURM job."
    )

    # Basic job info
    parser.add_argument("--job_name", required=True,
                        help="Base name for the job; a timestamp is appended automatically.")
    parser.add_argument("--template", default="slurm/templates/train_template.sbatch",
                        help="Path to the SLURM template .sbatch file.")
    parser.add_argument("--script", default="src/main.py",
                        help="Path to the Python script you want to run inside SLURM.")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to a config file (if your script requires one).")

    # SLURM resource options
    parser.add_argument("--time", default="01:00:00",
                        help="Requested runtime for the job.")
    parser.add_argument("--partition", default="gpu",
                        help="SLURM partition to submit to (e.g., 'gpu' or 'compute').")
    parser.add_argument("--gpus", default="1", 
                        help="Number of GPUs requested (if using GPUs).")
    parser.add_argument("--mem", default="16G",
                        help="Memory to allocate for the job.")
    parser.add_argument("--cpus", default="4",
                        help="Number of CPU cores per task.")

    # Output control
    parser.add_argument("--output", default="slurm/generated/job.sbatch",
                        help="Where to save the generated .sbatch file.")
    parser.add_argument("--submit", action="store_true",
                        help="If specified, automatically submit the job with 'sbatch'.")
    
    args = parser.parse_args()

    # Validate that required files exist
    validate_file_exists(args.template)
    validate_file_exists(args.script)
    validate_file_exists(args.config)

    # Load and fill template
    template_text = load_template(args.template)

    # Generate unique job name with timestamp
    job_name = generate_job_name(args.job_name)

    # Populate placeholders in the template
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

    # Save to the specified output path
    output_path = Path(args.output)
    save_script(filled_text, output_path)

    # If the user wants to automatically submit
    if args.submit:
        print(f"Submitting {output_path} to SLURM queue...")
        subprocess.run(["sbatch", str(output_path)])
    else:
        print(f"SLURM script generated at: {output_path}\n"
              f"Use 'sbatch {output_path}' to submit manually.")

if __name__ == "__main__":
    main()


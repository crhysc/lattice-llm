#!/usr/bin/env python3
import os
import argparse
import subprocess
from pathlib import Path
import yaml

from utils import (
    load_template,
    fill_template,
    generate_job_name,
    save_script,
    validate_file_exists
)

def generate_subcommand(args, defaults):
    """
    Handles 'generate' logic: fill a template, save it,
    and optionally submit if user wants to.
    """
    # Validate input files
    validate_file_exists(args.template)
    validate_file_exists(args.script)
    validate_file_exists(args.config)

    # Load the template
    template_text = load_template(args.template)

    # Generate job_name (required)
    job_name = generate_job_name(args.job_name)

    # Fill
    filled_text = fill_template(
        template_text,
        {
            "job_name": job_name,
            "script": args.script,
            "config": args.config,
            "time": args.time,
            "partition": args.partition,
            "gpus": args.gpus,
            "mem": args.mem,
            "cpus": args.cpus,
            "environment": args.environment,
        }
    )

    # Save
    output_path = Path(args.output)
    save_script(filled_text, output_path)
    print(f"Generated SLURM script: {output_path}")

    # Optionally submit
    if args.submit:
        print(f"Submitting {output_path} to SLURM queue...")
        subprocess.run(["sbatch", str(output_path)])


def submit_subcommand(args):
    """
    Handles 'submit' logic: submit an existing .sbatch file.
    """
    sbatch_file = Path(args.sbatch_file)
    validate_file_exists(sbatch_file)
    print(f"Submitting existing SBATCH script: {sbatch_file}")
    subprocess.run(["sbatch", str(sbatch_file)])


def main():
    # Try to load defaults
    defaults_file = Path(__file__).parent / "defaults.yaml"
    defaults = {}
    if defaults_file.exists():
        with open(defaults_file, "r") as f:
            defaults = yaml.safe_load(f)
        # Expand any environment variables in defaults
        for key, value in defaults.items():
            if isinstance(value, str):
                defaults[key] = os.path.expandvars(value)

    # Create top-level parser
    # Note the use of RawDescriptionHelpFormatter so that our epilog formatting is preserved
    parser = argparse.ArgumentParser(
        description="Generate or submit SLURM jobs.",
        epilog=(
            "-------------------- EXAMPLES --------------------\n"
            "Generate a new job script:\n"
            "  python make_job.py generate --job_name example_job \\\n"
            "      --template slurm/templates/train_template.sbatch \\\n"
            "      --script src/main.py --config config.yaml \\\n"
            "      --time 02:00:00 --partition gpu \\\n"
            "      --gpus 2 --mem 32G --cpus 8 \\\n"
            "      --environment my_conda_env \\\n"
            "      --output ./generated/my_job.sbatch --submit\n\n"
            "Submit an existing SBATCH script:\n"
            "  python make_job.py submit ./generated/my_job.sbatch\n"
            "---------------------------------------------------"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(
        title="subcommands",
        description="Valid subcommands",
        help="Use one of the commands below",
        dest="subcommand"
    )

    # ----------------------------------------------------------------------
    # 1) 'generate' subcommand
    # ----------------------------------------------------------------------
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate a new SLURM script from a template."
    )
    gen_parser.add_argument(
        "--job_name", required=True,
        help="Base name for the job; a timestamp is appended automatically."
    )
    gen_parser.add_argument(
        "--template",
        default=defaults.get("template", "slurm/templates/train_template.sbatch"),
        help="Path to the SLURM template .sbatch file."
    )
    gen_parser.add_argument(
        "--script",
        default=defaults.get("script", "src/main.py"),
        help="Path to the Python script you're running."
    )
    gen_parser.add_argument(
        "--config",
        default=defaults.get("config", "config.yaml"),
        help="Path to your config file."
    )
    # SLURM resource options
    gen_parser.add_argument("--time", default=defaults.get("time", "01:00:00"))
    gen_parser.add_argument("--partition", default=defaults.get("partition", "gpu"))
    gen_parser.add_argument("--gpus", default=defaults.get("gpus", "1"))
    gen_parser.add_argument("--mem", default=defaults.get("mem", "16G"))
    gen_parser.add_argument("--cpus", default=defaults.get("cpus", "4"))
    gen_parser.add_argument(
        "--environment",
        default=defaults.get("environment", None),
        help="Environment name to load in the job (e.g., conda env)."
    )
    gen_parser.add_argument(
        "--output",
        default=defaults.get("output", "./generated/job.sbatch"),
        help="Where to save the generated script."
    )
    gen_parser.add_argument(
        "--submit", action="store_true",
        help="If specified, automatically submit after generation."
    )
    gen_parser.set_defaults(func=lambda args: generate_subcommand(args, defaults))

    # ----------------------------------------------------------------------
    # 2) 'submit' subcommand
    # ----------------------------------------------------------------------
    sub_parser = subparsers.add_parser(
        "submit",
        help="Submit an already-generated SLURM script."
    )
    sub_parser.add_argument(
        "sbatch_file",
        help="Path to an existing SBATCH file to submit."
    )
    sub_parser.set_defaults(func=submit_subcommand)

    # ----------------------------------------------------------------------
    # Parse arguments
    # ----------------------------------------------------------------------
    args = parser.parse_args()

    # If user did not provide a subcommand, print help
    if not args.subcommand:
        parser.print_help()
        return

    # Dispatch
    args.func(args)


if __name__ == "__main__":
    main()


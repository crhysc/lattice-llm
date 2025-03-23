#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}.out
#SBATCH --error=logs/{job_name}.err
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}

# Activate environment
source ~/.bashrc
conda activate {environment}

# Run command
python {script} --config {config}


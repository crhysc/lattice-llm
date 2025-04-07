#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}.out
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}

mkdir -p logs

source ~/.bashrc
conda activate {{ environment }}

python {{ script }} --config {{ config }}


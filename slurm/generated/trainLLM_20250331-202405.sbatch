#!/bin/bash
#SBATCH --job-name=trainLLM_20250331-202405
#SBATCH --output=logs/trainLLM_20250331-202405.out
#SBATCH --time=07:00:00
#SBATCH --partition=gpu_7day
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

mkdir -p logs

source ~/.bashrc
conda activate { environment }

python { script } --config { config }


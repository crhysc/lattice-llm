#!/bin/bash
#SBATCH --job-name=download-data_20250327-210931
#SBATCH --output=logs/download-data_20250327-210931.out
#SBATCH --error=logs/download-data_20250327-210931.err
#SBATCH --time=07:00:00
#SBATCH --partition=comm_small_day
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

source ~/.bashrc
conda activate { environment }

bash { script }


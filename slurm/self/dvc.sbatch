#!/bin/bash
#SBATCH --job-name=dvc
#SBATCH --output=./output.out
#SBATCH --time=07:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate guadaloupe

cd ~/lattice-llm/
dvc commit


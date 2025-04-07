#!/bin/bash
#SBATCH --job-name=trainLLM
#SBATCH --output=./output.out
#SBATCH --time=07:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1

source ~/.bashrc
conda activate guadaloupe

cd ~/lattice-llm/
python ~/lattice-llm/src/main.py


#!/bin/bash
#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 96G
#SBATCH --time 12:00:00
#SBATCH --job-name averitec
#SBATCH --output logs/jupyter.%j.out

ml Python/3.10.4-GCCcore-11.3.0-bare

# Replace with your own virtual environment
source ~/aic_averitec/venvs/averitec/bin/activate

# load your .env
set -o allexport
source .env
set +o allexport


python3 src/run_vllm_api.py

#!/bin/bash
#SBATCH --partition amdgpufast
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 96G
#SBATCH --gres gpu:1
#SBATCH --time 4:00:00
#SBATCH --job-name averitec
#SBATCH --output logs/jupyter.%j.out

ml Python

# Replace with your own virtual environment
source ~/my_env/bin/activate

# Replace with absolute path to your project or sbatch from project root and comment out
# cd ~/ullriher/aic_averitec

# load your .env
source .env

export PYTHONPATH=src:$PYTHONPATH
jupyter notebook --no-browser --port=$(shuf -i8000-9999 -n1) --ip=$(hostname -s)

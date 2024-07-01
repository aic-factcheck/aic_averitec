#!/bin/bash
#SBATCH --partition amdgpufast
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 96G
#SBATCH --gres gpu:1
#SBATCH --time 4:00:00
#SBATCH --job-name averitec
#SBATCH --output /home/ullriher/ullriher/logs/jupyter/averitec.%j.out

ml Python/3.10.4-GCCcore-11.3.0-bare

# Replace with your own virtual environment
source ~/venvs/2023feb/bin/activate

# Replace with absolute path to your project
cd ~/ullriher/aic_averitec

export PYTHONPATH=src:$PYTHONPATH
jupyter notebook --no-browser --port=$(shuf -i8000-9999 -n1) --ip=$(hostname -s)

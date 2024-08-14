#!/bin/bash
#SBATCH --partition amdgpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 96G
#SBATCH --gres gpu:2
#SBATCH --time 12:00:00
#SBATCH --job-name averitec_vllm
#SBATCH --output logs/jupyter.%j.out


ml Python/3.10.4-GCCcore-11.3.0-bare

# Replace with your own virtual environment
source ~/aic_averitec/venvs/vllm/bin/activate

export PYTHONPATH=src:$PYTHONPATH
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

#serve vllm
vllm serve TechxGenus/Mistral-Large-Instruct-2407-AWQ --model TechxGenus/Mistral-Large-Instruct-2407-AWQ --port 8095 --gpu-memory-utilization 0.95 --max-model-len 128000 --tensor-parallel-size 2 --api-key token-abc123
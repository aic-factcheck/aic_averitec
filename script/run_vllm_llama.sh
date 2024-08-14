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

#serve vllm
vllm serve hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 --port 8094 --gpu-memory-utilization 0.95 --max-model-len 65536 --tensor-parallel-size 2 --api-key token-abc123
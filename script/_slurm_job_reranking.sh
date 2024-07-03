#!/bin/bash
#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 96G
#SBATCH --gres gpu:1
#SBATCH --time 24:00:00
#SBATCH --job-name averitec_reranking
#SBATCH --output /home/mlynatom/logs/averitec_reranking.%j.out


ml PyTorch/2.3.0-foss-2023b-CUDA-12.4.0

source /home/mlynatom/venvs/05_2024_py3.11.5/bin/activate

export PATH=/home/mlynatom/venvs/05_2024_py3.11.5/bin:${PATH}
export PYTHONPATH=/home/mlynatom/experimental-mlynatom/src:${PYTHONPATH}


python aic_averitec/src/reranking/rerank_sentences.py
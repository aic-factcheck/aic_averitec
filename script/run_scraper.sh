#!/bin/bash
#SBATCH --partition amd
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 96G
#SBATCH --time 24:00:00
#SBATCH --job-name averitec_scraping
#SBATCH --output /home/mlynatom/logs/averitec_scraping.%j.out


ml PyTorch/2.3.0-foss-2023b-CUDA-12.4.0

source /home/mlynatom/venvs/05_2024_py3.11.5/bin/activate

# pip uninstall fitz
# pip install pymupdf
pip install trafilatura
pip install spacy

export PATH=/home/mlynatom/venvs/05_2024_py3.11.5/bin:${PATH}
export PYTHONPATH=/home/mlynatom/experimental-mlynatom/src:${PYTHONPATH}


bash /home/mlynatom/aic_averitec/script/scraper.sh dev 0 500
#!/bin/bash
#SBATCH --partition interactive
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 128G
#SBATCH --gres gpu:1
#SBATCH --time 72:00:00
#SBATCH --job-name averitec
#SBATCH --output logs/jupyter.%j.out
unset LMOD_ROOT; unset MODULESHOME; unset LMOD_PKG; unset LMOD_CMD; unset LMOD_DIR; unset FPATH; unset __LMOD_REF_COUNT_MODULEPATH; unset __LMOD_REF_COUNT__LMFILES_; unset _LMFILES_; unset _ModuleTable001_; unset _ModuleTable002_
source /etc/profile.d/lmod.sh
module load Python/3.10.4-GCCcore-11.3.0-bare

# Replace with your own virtual environment
source ~/venvs/averitec/bin/activate

# Replace with absolute path to your project or sbatch from project root and comment out
# cd ~/ullriher/aic_averitec

# load your .env
source .env

export PYTHONPATH=src:$PYTHONPATH
jupyter notebook --no-browser --port=$(shuf -i8000-9999 -n1) --ip=$(hostname -s)

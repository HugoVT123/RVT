#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=05:00:00
#SBATCH --partition=vicomtech
#SBATCH --qos=qos_di11
#SBATCH --output=messages/preprocess_DSEC-%j.out
#SBATCH --error=messages/preprocess_DSEC-%j.err
#SBATCH --job-name=preprocess_DSEC
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hsosapavon@vicomtech.org
#SBATCH --mem=170GB
#SBATCH --gres=gpu:1
#SBATCH --nodes=1


module purge
module load Miniconda3/23.10.0-1
module load cuDNN/9.1.1.17-CUDA-12.4.0

source activate rvt

python preprocess_dsec.py

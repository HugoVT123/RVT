#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --partition=vicomtech
#SBATCH --qos=qos_di11
#SBATCH --output=messages/preprocess_dsec-%j.out
#SBATCH --error=messages/preprocess_dsec_job-%j.err
#SBATCH --job-name=preprocess_dsec
#SBATCH --mail-type=NONE
#SBATCH --mail-user=hsosapavon@vicomtech.org
#SBATCH --mem=192GB
#SBATCH --gres=gpu:1 -C "t4"
#SBATCH --nodes=1


module purge
module load Miniconda3/23.10.0-1
module load cuDNN/9.1.1.17-CUDA-12.4.0

source activate rvt

python preprocess_dsec.py

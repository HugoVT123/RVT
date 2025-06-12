#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=10:00:00
#SBATCH --partition=vicomtech
#SBATCH --qos=qos_di11
#SBATCH --output=messages/preprocess_dsec_v3-%j.out
#SBATCH --error=messages/preprocess_dsec_v3-%j.err
#SBATCH --job-name=preprocess_dsec_v3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hsosapavon@vicomtech.org
#SBATCH --mem=180GB
#SBATCH --gres=gpu:1 -C "a40"
#SBATCH --nodes=1


module purge
module load Miniconda3/23.10.0-1
module load cuDNN/9.1.1.17-CUDA-12.4.0

source activate rvt

python preprocess_dsec_v3.py

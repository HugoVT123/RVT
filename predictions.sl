#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=10:00:00
#SBATCH --partition=vicomtech
#SBATCH --qos=qos_di11
#SBATCH --output=messages/predictions-%j.out
#SBATCH --error=messages/predictions-%j.err
#SBATCH --job-name=predictions
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hsosapavon@vicomtech.org
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1 -C "t4"
#SBATCH --nodes=1


module purge
module load Miniconda3/23.10.0-1
module load cuDNN/9.1.1.17-CUDA-12.4.0

source activate rvt

python predictions.py

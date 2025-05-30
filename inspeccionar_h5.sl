#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --partition=vicomtech
#SBATCH --qos=qos_di11
#SBATCH --output=messages/inspeccionar_h5-%j.out
#SBATCH --error=messages/inspeccionar_h5_job-%j.err
#SBATCH --job-name=inspeccionar_h5
#SBATCH --mail-type=NONE
#SBATCH --mail-user=hsosapavon@vicomtech.org
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --nodes=1



module purge
module load Miniconda3/23.10.0-1
module load cuDNN/9.1.1.17-CUDA-12.4.0

source activate rvt

python inspeccionar_h5.py
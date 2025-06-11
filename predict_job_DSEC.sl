#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=vicomtech
#SBATCH --qos=qos_di11
#SBATCH --output=messages/predict-DSEC-%j.out
#SBATCH --error=messages/predict_DSEC-%j.err
#SBATCH --job-name=predict
#SBATCH --mail-type=NONE
#SBATCH --mail-user=hsosapavon@vicomtech.org
#SBATCH --mem=192GB
#SBATCH --gres=gpu:1 -C "a40"
#SBATCH --nodes=1



module purge
module load Miniconda3/23.10.0-1
module load cuDNN/9.1.1.17-CUDA-12.4.0

source activate rvt

python predict.py checkpoint=checkpoints/rvt-b-gen4.ckpt dataset=gen4 dataset.path=data/dsec_proc use_test_set=0 hardware.gpus=0 +experiment/gen4=base.yaml hardware.num_workers.eval=1 batch_size.eval=1 model.postprocess.confidence_threshold=0.001


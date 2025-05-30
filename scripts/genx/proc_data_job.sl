#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --partition=vicomtech
#SBATCH --qos=qos_di11
#SBATCH --output=messages/proc_data%j.out
#SBATCH --error=messages/proc_data%j.err
#SBATCH --job-name=proc_data
#SBATCH --mail-type=NONE
#SBATCH --mail-user=hsosapavon@vicomtech.org
#SBATCH --mem=16GB
#SBATCH --nodes=1

module purge
module load Miniconda3/23.10.0-1

source activate rvt

mkdir -p messages

python preprocess_dataset.py ../../data/dummy_dsec ../../data/dummy_dsec_proc conf_preprocess/representation/stacked_hist.yaml conf_preprocess/extraction/const_duration.yaml conf_preprocess/filter_gen4.yaml -ds gen4 -np 20
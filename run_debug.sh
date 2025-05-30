#!/bin/bash

# Step 1: Request GPU node
salloc --ntasks=1 --mem=16G --gres=gpu:1 --time=00:10:00 <<EOF
  # Step 2: Activate environment
  source ~/.bashrc
  conda activate rvt

  # Step 3: Run script (which contains pydevd_pycharm.settrace())
  python predict.py dataset=gen1 dataset.path=data/dummy_gen1 checkpoint=checkpoints/rvt-b.ckpt use_test_set=0 hardware.gpus=0 +experiment/gen1="base.yaml" hardware.num_workers.eval=1 batch_size.eval=1 model.postprocess.confidence_threshold=0.001
EOF


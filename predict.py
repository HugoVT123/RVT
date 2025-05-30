
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelSummary
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf

from config.modifier import dynamically_modify_train_config
from modules.utils.fetch import fetch_model_module, fetch_data_module
from data.utils.types import ObjDetOutput

# Desired dtype
dtype = np.dtype({
    'names': ['t', 'x', 'y', 'w', 'h', 'class_id', 'track_id', 'class_confidence'],
    'formats': ['<i8', '<f4', '<f4', '<f4', '<f4', '<u4', '<u4', '<f4'],
    'offsets': [0, 8, 12, 16, 20, 24, 28, 32],
    'itemsize': 40
})


@hydra.main(config_path='config', config_name='val', version_base='1.2')
def main(config: DictConfig):
    # Modifica dinámicamente el config
    dynamically_modify_train_config(config)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    # --- Load model and data module ---
    data_module = fetch_data_module(config=config)
    data_module.setup('validate')

    ckpt_path = Path(config.checkpoint)
    module = fetch_model_module(config=config)
    module = module.load_from_checkpoint(str(ckpt_path), **{'full_config': config})

    # --- Setup logger and callbacks ---
    logger = CSVLogger(save_dir='./prediction_logs')
    callbacks = [ModelSummary(max_depth=2)]

    # --- Define GPU/CPU usage ---
    use_gpu = torch.cuda.is_available()
    accelerator = 'gpu' if use_gpu else 'cpu'
    devices = [config.hardware.gpus] if use_gpu else 1

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=callbacks,
        precision=config.training.precision,
        move_metrics_to_cpu=False,
    )

    with torch.inference_mode():
        data = trainer.predict(
            model=module,
            dataloaders=data_module.val_dataloader(),
            ckpt_path=str(ckpt_path),
            return_predictions=True,
        )

        
        # Let's say you want to look at the first item
        print(type(data))  # Should be a list or similar structure
        print(len(data))  # Should be the number of predictions

        all_preds = []

        for frame_prediction in data:
            predictions = frame_prediction[ObjDetOutput.PRED_PROPH]
            print(frame_prediction[ObjDetOutput.LABELS_PROPH])

            for bbox in predictions:
                entry = (
                    int(bbox['t']),
                    float(bbox['x']),
                    float(bbox['y']),
                    float(bbox['w']),
                    float(bbox['h']),
                    int(bbox['class_id']),
                    int(bbox['track_id']),
                    float(bbox['class_confidence']),
                )
                all_preds.append(entry)

        # Convert to structured NumPy array with the specified dtype
        np_array = np.array(all_preds, dtype=dtype)

        # Save to .npz file
        output_file = "predictions/predictions.npz"
        np.savez_compressed(output_file, predictions=np_array)
        print(f"Predictions saved to {output_file}")
   

if __name__ == "__main__":
    main()

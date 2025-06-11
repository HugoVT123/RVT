
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
import shutil
from pathlib import Path
from copy import deepcopy

# Desired dtype
dtype = np.dtype({
    'names': ['t', 'x', 'y', 'w', 'h', 'class_id', 'track_id', 'class_confidence'],
    'formats': ['<i8', '<f4', '<f4', '<f4', '<f4', '<u4', '<u4', '<f4'],
    'offsets': [0, 8, 12, 16, 20, 24, 28, 32],
    'itemsize': 40
})

@hydra.main(config_path='config', config_name='val', version_base='1.2')
def main(config: DictConfig):
    original_dataset_root = Path(config.dataset.path)
    test_dir = original_dataset_root / "val"
    all_sequences = sorted([d for d in test_dir.iterdir() if d.is_dir()])

    for sequence_path in all_sequences:
        print(f"\n=== Procesando secuencia: {sequence_path.name} ===")

        # 1. Crear una copia temporal del dataset
        tmp_dataset_root = original_dataset_root.parent / "dsec_proc_tmp"
        if tmp_dataset_root.exists():
            shutil.rmtree(tmp_dataset_root)
        shutil.copytree(original_dataset_root, tmp_dataset_root)

        # 2. Dejar solo esta secuencia en `test/`
        tmp_test_dir = tmp_dataset_root / "val"
        for other_seq in tmp_test_dir.iterdir():
            if other_seq.name != sequence_path.name:
                if other_seq.is_dir():
                    shutil.rmtree(other_seq)
                else:
                    other_seq.unlink()

        # 3. Actualizar config con la ruta temporal
        local_config = deepcopy(config)
        local_config.dataset.path = str(tmp_dataset_root)

        # 4. Ejecutar predicción como antes
        run_prediction_for_sequence(local_config, sequence_path.name)

        # 5. Eliminar la copia temporal
        shutil.rmtree(tmp_dataset_root)
        print(f"✔ Secuencia {sequence_path.name} procesada y limpia.")

def run_prediction_for_sequence(config: DictConfig, sequence_name: str):
    dynamically_modify_train_config(config)

    print('------ Configuración actual ------')
    print(OmegaConf.to_yaml(config))
    print('----------------------------------')

    data_module = fetch_data_module(config=config)
    data_module.setup('validate')

    ckpt_path = Path(config.checkpoint)
    module = fetch_model_module(config=config)
    module = module.load_from_checkpoint(str(ckpt_path), **{'full_config': config})

    logger = CSVLogger(save_dir=f'./prediction_logs/{sequence_name}')
    callbacks = [ModelSummary(max_depth=2)]

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

        all_preds = []
        for idx, frame_prediction in enumerate(data):
            if ObjDetOutput.PRED_PROPH not in frame_prediction:
                print(f"[WARNING] Frame {idx} missing PRED_PROPH. Keys: {frame_prediction.keys()}")
                continue

            predictions = frame_prediction[ObjDetOutput.PRED_PROPH]
            for bbox in predictions:
                print(f"Timestamp: {bbox['t']}")  # Debugging
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

        np_array = np.array(all_preds, dtype=dtype)
        os.makedirs("predictions", exist_ok=True)
        output_file = f"predictions/{sequence_name}.npz"
        np.savez_compressed(output_file, predictions=np_array)
        print(f"✔ Predicciones guardadas en {output_file}") 

if __name__ == "__main__":
    main()

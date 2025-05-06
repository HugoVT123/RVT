import os
import cv2
import numpy as np
from pathlib import Path
import torch
from torch.backends import cuda, cudnn

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy('file_system')

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelSummary

from config.modifier import dynamically_modify_train_config
from modules.utils.fetch import fetch_data_module, fetch_model_module
from callbacks.custom import get_viz_callback
from callbacks.viz_base import VizCallbackBase
from utils.evaluation.prophesee.visualize.vis_utils import draw_bboxes_bbv
from data.utils.types import DataType
import torch.nn.functional as F

dtype = np.dtype([
    ('x', 'f4'), ('y', 'f4'), ('w', 'f4'), ('h', 'f4'),
    ('class_id', 'i4'), ('class_confidence', 'f4')
])

@hydra.main(config_path='config', config_name='val', version_base='1.2')
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    # ---------------------
    # GPU options
    # ---------------------
    gpus = config.hardware.gpus
    assert isinstance(gpus, int), 'no more than 1 GPU supported'
    gpus = [gpus]

    # ---------------------
    # Data
    # ---------------------
    data_module = fetch_data_module(config=config)

    # ---------------------
    # Logging and Checkpoints
    # ---------------------
    logger = CSVLogger(save_dir='./validation_logs')
    ckpt_path = Path(config.checkpoint)

    # ---------------------
    # Model
    # ---------------------

    module = fetch_model_module(config=config)
    module = module.load_from_checkpoint(str(ckpt_path), **{'full_config': config})

    # ---------------------
    # Callbacks and Misc
    # ---------------------
    viz_callback = get_viz_callback(config)
    callbacks = [ModelSummary(max_depth=2), viz_callback]

    # ---------------------
    # Validation
    # ---------------------

    trainer = pl.Trainer(
        accelerator='gpu',
        callbacks=callbacks,  # Use the full list of callbacks
        default_root_dir=None,
        devices=gpus,
        logger=logger,
        log_every_n_steps=1,
        precision=config.training.precision,
        move_metrics_to_cpu=False,
    )

    # Perform validation and draw bounding boxes
    output_dir = Path("output/validation_bboxes")
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        if config.use_test_set:
            results = trainer.test(model=module, datamodule=data_module, ckpt_path=str(ckpt_path))
        else:
            results = trainer.validate(model=module, datamodule=data_module, ckpt_path=str(ckpt_path))

        # Iterate over validation data and draw bounding boxes
        print(results)
        dataloader = data_module.val_dataloader()
        for batch_idx, batch in enumerate(dataloader):
            
            # Access event representation using DataType enum
            event_repr = batch["data"][DataType.EV_REPR]  # Event representation

            # Iterate over the batch
            for seq_idx, ev_repr in enumerate(event_repr):
                # Remove the batch dimension and convert event representation to an image
                print(f"Processing batch {batch_idx}, sequence {seq_idx}")
                print(f"Shape of ev_repr before unsqueeze: {ev_repr.shape}")
                print(f"Data type of ev_repr before conversion: {ev_repr.dtype}")
            
                ev_repr = ev_repr.to(torch.float)
                ev_img = VizCallbackBase.ev_repr_to_img(ev_repr.squeeze(0).cpu().numpy())
                rescaled_ev_img = cv2.resize(ev_img, (ev_img.shape[1] * 4, ev_img.shape[0] * 4), interpolation=cv2.INTER_AREA)

                # Get predictions from the model
                with torch.no_grad():
                    print(f"Shape of ev_repr before passing to the model: {ev_repr.shape}")
                    print(f"Data type of ev_repr before conversion: {ev_repr.dtype}")
                    
                    # Convert to the correct data type (torch.float)
                    if ev_repr.dtype != torch.float:
                        ev_repr = ev_repr.to(torch.float)
                        print(f"Data type of ev_repr after conversion: {ev_repr.dtype}")
                    
                    # Normalize the input if necessary
                    ev_repr = ev_repr / 255.0  # Normalize to [0, 1] if required by the model

                    # Pad the input tensor to make its dimensions divisible by the window size
                    window_size = 8  # Replace with the actual window size used by your model
                    height, width = ev_repr.shape[-2], ev_repr.shape[-1]
                    pad_h = (window_size - height % window_size) % window_size
                    pad_w = (window_size - width % window_size) % window_size

                    # Apply padding
                    ev_repr = F.pad(ev_repr, (0, pad_w, 0, pad_h))  # Pad (left, right, top, bottom)
                    print(f"Shape of ev_repr after padding: {ev_repr.shape}")

                    # Pass the tensor to the model
                    predictions, _, _ = module(ev_repr.to(module.device))  # Move to device

                # Post-process predictions
                pred_processed = postprocess(
                    prediction=predictions,
                    num_classes=config.model.head.num_classes,
                    conf_thre=config.model.postprocess.confidence_threshold,
                    nms_thre=config.model.postprocess.nms_threshold
                )

                # Convert predictions to a structured array
                if pred_processed is not None:
                    structured_labels = np.zeros(len(pred_processed), dtype=dtype)
                    structured_labels['x'] = pred_processed[:, 0]  # x1
                    structured_labels['y'] = pred_processed[:, 1]  # y1
                    structured_labels['w'] = pred_processed[:, 2] - pred_processed[:, 0]  # width
                    structured_labels['h'] = pred_processed[:, 3] - pred_processed[:, 1]  # height
                    structured_labels['class_id'] = pred_processed[:, 6].astype('i4')  # class ID
                    structured_labels['class_confidence'] = pred_processed[:, 5]  # confidence score

                    # Draw bounding boxes
                    rescaled_ev_img = draw_bboxes_bbv(rescaled_ev_img, structured_labels)

                # Save the image
                seq_output_dir = output_dir / f"sequence_{batch_idx}"
                seq_output_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(seq_output_dir / f"frame_{seq_idx}.png"), rescaled_ev_img)



if __name__ == '__main__':
    main()

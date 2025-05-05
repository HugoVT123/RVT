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
            
            # Access event representation and labels using DataType enum
            event_repr = batch["data"][DataType.EV_REPR]  # Event representation
            labels = batch["data"][DataType.OBJLABELS_SEQ]  # Ground truth labels (SparselyBatchedObjectLabels)

            # Check if event representation is a list
            # Iterate over the batch
            for seq_idx, (ev_repr, sparse_label) in enumerate(zip(event_repr, labels)):
                # Remove the batch dimension and convert event representation to an image
                print(f"Processing batch {batch_idx}, sequence {seq_idx}")
            
                ev_img = VizCallbackBase.ev_repr_to_img(ev_repr.squeeze(0).cpu().numpy())

                rescaled_ev_img = cv2.resize(ev_img, (ev_img.shape[1] * 4, ev_img.shape[0] * 4), interpolation=cv2.INTER_AREA)

                # Check if the sparse_label is not None
                if sparse_label is not None:
                    
                    # Extract valid ObjectLabels from the sparse label
                    for obj_label in sparse_label:
                        if obj_label is not None:
                            print(obj_label.x, obj_label.y, obj_label.w, obj_label.h, obj_label.class_id, obj_label.class_confidence)
                            
                            # Convert ObjectLabels to a tensor
                            label_array = obj_label.get_labels_as_tensors().cpu().numpy()

                            

                            # Create a structured array
                            structured_labels = np.zeros(label_array.shape[0], dtype=[
                                ('x', 'f4'), ('y', 'f4'), ('w', 'f4'), ('h', 'f4'),
                                ('class_id', 'i4'), ('class_confidence', 'f4')
                            ])

                            # Map the fields correctly
                            structured_labels['class_id'] = label_array[:, 0].astype('i4')  # Object ID
                            structured_labels['x'] = label_array[:, 1]  # Height
                            structured_labels['y'] = label_array[:, 2]  # Width
                            structured_labels['w'] = label_array[:, 3]  # Y-coordinate
                            structured_labels['h'] = label_array[:, 4]  # X-coordinate

                            structured_labels['x'] = structured_labels['x'] - (structured_labels['w'] / 2)
                            structured_labels['y'] = structured_labels['y'] - (structured_labels['h'] / 2)


                            # Draw bounding boxes
                            rescaled_ev_img=draw_bboxes_bbv(ev_img, structured_labels)

                # Save the image
                seq_output_dir = output_dir / f"sequence_{batch_idx}"
                seq_output_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(seq_output_dir / f"frame_{seq_idx}.png"), rescaled_ev_img)



if __name__ == '__main__':
    main()

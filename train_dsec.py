import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from modules.detection import Module
from config.modifier import dynamically_modify_train_config
import os
from pytorch_lightning.strategies import DDPStrategy

def train():
    """
    Main training function.
    """
    # -------------------------------
    # 1. Configuration
    # -------------------------------
    with initialize(config_path="config"):
        config = compose(config_name="train", overrides=[
            "+experiment/gen4=base.yaml",
            "dataset=gen4",
            "dataset.path=data/dsec_proc",
            "hardware.gpus=1",
            "batch_size=4",
            "hardware.num_workers=4",
            "training.learning_rate=0.001",
            # Add any other training-specific overrides here
            "wandb.group_name=my_training_run"
        ])

    dynamically_modify_train_config(config)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    # -------------------------------
    # 2. Model Initialization
    # -------------------------------
    # The Module's __init__ will automatically handle setting up the model,
    # datasets, and dataloaders from the config.
    model = Module(full_config=config)

    # -------------------------------
    # 3. Callbacks and Logger
    # -------------------------------
    # Create a logger
    logger = TensorBoardLogger("tb_logs", name="dsec_training")

    # Create a checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints/dsec_training",
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    gpu_config = config.hardware.gpus
    gpus = OmegaConf.to_container(gpu_config) if OmegaConf.is_config(gpu_config) else gpu_config
    gpus = gpus if isinstance(gpus, list) else [gpus]
    distributed_backend = config.hardware.dist_backend
    assert distributed_backend in ('nccl', 'gloo'), f'{distributed_backend=}'
    strategy = DDPStrategy(process_group_backend=distributed_backend,
                           find_unused_parameters=False,
                           gradient_as_bucket_view=True) if len(gpus) > 1 else None

    distributed_backend = config.hardware.dist_backend
    strategy = DDPStrategy(process_group_backend=distributed_backend,
                           find_unused_parameters=False,
                           gradient_as_bucket_view=True) if len(gpus) > 1 else None

    # -------------------------------
    # 4. Trainer Initialization and Training
    # -------------------------------
    trainer = pl.Trainer(
        accelerator='gpu',
        enable_checkpointing=True,
        default_root_dir=None,
        devices=gpus,
        gradient_clip_val=config.training.gradient_clip_val,
        gradient_clip_algorithm='value',
        limit_train_batches=config.training.limit_train_batches,
        limit_val_batches=config.validation.limit_val_batches,
        logger=logger,
        log_every_n_steps=config.logging.train.log_every_n_steps,
        plugins=None,
        precision=config.training.precision,
        max_epochs=config.training.max_epochs,
        max_steps=config.training.max_steps,
        strategy=strategy,
        sync_batchnorm=False if strategy is None else True,
        move_metrics_to_cpu=False,
        benchmark=config.reproduce.benchmark,
        deterministic=config.reproduce.deterministic_flag,
    )

    # Start the training process
    # The trainer will automatically handle the training and validation loops.
    trainer.fit(model)

if __name__ == "__main__":
    train()
import torch
import h5py
import numpy as np
import cv2
import torch.nn.functional as F
from omegaconf import OmegaConf
from hydra import compose, initialize
from config.modifier import dynamically_modify_train_config
from modules.detection import Module
from models.detection.yolox.utils import postprocess
import os

dtype = np.dtype([
    ('frame_idx', np.int32),
    ('class_id', np.int8),
    ('bbox', np.float32, (4,)),
    ('confidence', np.float32)
])


with initialize(config_path="config"):
    config = compose(config_name="val", overrides=[
        "+experiment/gen4=base.yaml",
        "checkpoint=checkpoints/rvt-b-gen4.ckpt",
        "dataset=gen4",
        "dataset.path=data/dsec_proc",
        "use_test_set=0",
        "hardware.gpus=0",
        "batch_size=1", # Note: This batch_size in config might be for validation/test loading,
                        # not necessarily the BATCH_SIZE you define below.
        "hardware.num_workers=0",
        "model.postprocess.confidence_threshold=0.001"
    ])

dynamically_modify_train_config(config)
OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

ckpt_path = config.checkpoint
model = Module.load_from_checkpoint(ckpt_path, full_config=config)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------------
# Load Event Representation
# -------------------------------
root_dir = "data/dsec_proc/test"
all_sequences = [item for item in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, item))]

all_sequences = ["interlaken_00_b"] # DELETE BEFORE DOING AUTOMATIZATION


for sequence in all_sequences:
    print(f"Processing sequence: {sequence}")
    h5_path = os.path.join(root_dir, sequence, "event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations_ds2_nearest.h5")
    predictions_path = os.path.join("predictions",f"{sequence}.npy")

    # -------------------------------
    # Load and Process Event Frames in Batches (streamed)
    # -------------------------------
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        print("Available datasets:", keys)
        dataset = f[keys[0]]  # Shape: (T, C, H, W)
        num_frames = dataset.shape[0]
        print(f"Total frames in dataset: {num_frames}")

        BATCH_SIZE = 1 # This batch size applies to the number of frames processed at once by model.forward
        TARGET_H = 384
        pred_processed_all = []
        all_predictions = []

        # --- IMPORTANT MODIFICATION FOR RECURRENCE ---
        # Initialize the recurrent states for the start of a new sequence.
        # The `Module`'s `_val_test_step_impl` uses `self.mode_2_rnn_states[mode].reset(...)`
        # and then `get_states(...)`.
        # For this external script, we need to directly manage the `LstmStates` tuple (c, h).
        # When `previous_states` is None, the `YoloXDetector`'s `forward_backbone`
        # (which is called by Module.forward when previous_states is None)
        # likely initializes the states to zeros.
        # So, for the first batch of a sequence, we pass None.
        current_rnn_states = None # This will hold the (c,h) states for all stages

        for start_idx in range(0, num_frames, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, num_frames)

            # Load only the batch slice from HDF5
            batch_np = dataset[start_idx:end_idx]  # shape: (B, C, H, W) where B = actual batch size (<= BATCH_SIZE)
            batch_tensor = torch.FloatTensor(batch_np).to(device)

            # Resize
            batch_resized = F.interpolate(batch_tensor, size=(TARGET_H, batch_tensor.shape[3]), mode='bicubic', align_corners=False)

            with torch.no_grad():
                # Pass the current states, and get the new states
                # The model's forward method returns (predictions, losses/None, new_states)
                preds, _, new_rnn_states = model.forward(batch_resized, previous_states=current_rnn_states)
                
                # Update the states for the next iteration
                current_rnn_states = new_rnn_states

            preds_post = postprocess(
                prediction=preds,
                num_classes=3,
                conf_thre=0.1,
                nms_thre=0.45
            )

            # pred_processed_all will accumulate a list of lists of detections.
            # Each inner list corresponds to one frame in the batch.
            pred_processed_all.extend(preds_post)

        # --- Corrected loop for storing detections with absolute frame_idx ---
        # Iterate over the accumulated detections for the entire sequence
        for i, detections_for_frame in enumerate(pred_processed_all):
            # `detections_for_frame` is the list of detections for the i-th frame in the sequence
            
            # Only process if detections exist for this frame
            if detections_for_frame is not None:
                for obj in detections_for_frame:
                    x1, y1, x2, y2, obj_conf, class_conf, class_pred = obj
                    scale_y = 360/384 # Original height was 360, target H is 384 based on your viz code,
                                      # but detection output will be scaled to original H 360 if downsampled in viz.
                                      # Check if `scale_y` is needed for the output bounding box format
                                      # or if `postprocess` already gives detections at original scale.
                                      # Given your `visualize_event_tensor_video_with_predictions` uses it,
                                      # it's likely correct here too.
                    
                    x1, y1, x2, y2 = float(x1), float(y1*scale_y), float(x2), float(y2*scale_y)

                    confidence = float(class_conf)
                    class_pred = int(class_pred)

                    all_predictions.append({
                        "frame_idx": i, # 'i' is now the correct absolute frame index within the sequence
                        "class_id": class_pred,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence # Use class_conf, not obj_conf here, as you did before.
                    })
            # else: # If no detections for a frame, you might want to record that (e.g., an empty list)
            #     all_predictions.append({
            #         "frame_idx": i,
            #         "class_id": -1, # Or some indicator that no objects were found
            #         "bbox": [0,0,0,0],
            #         "confidence": 0.0
            #     })

        structured_array = np.array([
            (p["frame_idx"], p["class_id"], p["bbox"], p["confidence"])
            for p in all_predictions
        ], dtype=dtype)

        os.makedirs(os.path.dirname(predictions_path), exist_ok=True) # Ensure directory exists
        np.savez_compressed(predictions_path.replace(".npy", ".npz"), predictions=structured_array)
        print(f"Saved predictions for sequence {sequence} to {predictions_path.replace('.npy', '.npz')}")

    # -------------------------------
    # Create Video with Bounding Boxes (Optional, if you want to visualize immediately)
    # -------------------------------
    # This call uses the `pred_processed_all` list that was just populated.
    # Note: If `pred_processed_all` is very large, consider visualizing in batches or separately.
    #visualize_event_tensor_video_with_predictions(h5_path, pred_processed_all, output_dir= os.path.join("visuals",sequence), max_frames=num_frames, downsample=True) # Use num_frames to visualize all
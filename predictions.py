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


def visualize_event_tensor_video_with_predictions(h5_path,pred_processed_all,output_dir='visuals', max_frames=100, downsample=True):
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        data = f['/data']  # Shape: (N, 2*T, H, W)
        print(f"Data shape: {data.shape}")
        total_frames = data.shape[0]
        num_frames = total_frames
        zero_count = 0

        print(f"Total frames in file: {total_frames}")
        print(f"Visualizing up to {num_frames} frames...")

        for i in range(num_frames):
            frame = data[i]  # Shape: (2*T, H, W)
            T = frame.shape[0] // 2
            pos = np.sum(frame[:T], axis=0)
            neg = np.sum(frame[T:], axis=0)

            if np.all(pos == 0) and np.all(neg == 0):
                zero_count += 1
                continue

            if downsample:
                pos = cv2.resize(pos, (640, 360), interpolation=cv2.INTER_NEAREST)
                neg = cv2.resize(neg, (640, 360), interpolation=cv2.INTER_NEAREST)

            # Normalize to 0–255
            pos_norm = cv2.normalize(pos, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            neg_norm = cv2.normalize(neg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Create RGB image: Red = positive events, Blue = negative events
            rgb = np.zeros((pos.shape[0], pos.shape[1], 3), dtype=np.uint8)
            rgb[..., 2] = pos_norm  # Red channel
            rgb[..., 0] = neg_norm  # Blue channel

            detections = pred_processed_all[i]

            # Only draw boxes if detections exist
            if detections is not None:
                for obj in detections:
                    x1, y1, x2, y2, obj_conf, class_conf, class_pred = obj
                    scale_y = 360/384 
                    x1, x2, y1, y2 = int(x1), int(x2), int(y1*scale_y), int(y2*scale_y)
                    h = y2 - y1
                    

                    if class_conf > 0.85:
                        cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(rgb, f"Class: {int(class_pred)}, Conf: {class_conf:.2f}",
                                    (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                print(f"No detections for frame {i}")

            # Save image
            cv2.imwrite(os.path.join(output_dir, f'frame_{i:04d}.png'), rgb)

        print(f"\n{zero_count} of {num_frames} frames are completely empty.")

# -------------------------------
# Configuration & Model Loading
# -------------------------------
with initialize(config_path="config"):
    config = compose(config_name="val", overrides=[
        "+experiment/gen4=default.yaml",
        "checkpoint=checkpoints/rvt-b-gen4.ckpt",
        "dataset=gen4",
        "dataset.path=data/1mpx_proc",
        "use_test_set=0",
        "hardware.gpus=0",
        "batch_size=1",
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
sequence = "thun_01_b"
h5_path = os.path.join(root_dir, sequence, "event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations_ds2_nearest.h5")

# -------------------------------
# Load and Process Event Frames in Batches (streamed)
# -------------------------------
with h5py.File(h5_path, "r") as f:
    keys = list(f.keys())
    print("Available datasets:", keys)
    dataset = f[keys[0]]  # Shape: (T, C, H, W)
    num_frames = dataset.shape[0]
    print(f"Total frames in dataset: {num_frames}")

    BATCH_SIZE = 50
    TARGET_H = 384
    pred_processed_all = []

    for start_idx in range(0, num_frames, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, num_frames)

        # Load only the batch slice from HDF5
        batch_np = dataset[start_idx:end_idx]  # shape: (B, C, H, W)
        batch_tensor = torch.FloatTensor(batch_np).to(device)

        # Resize
        batch_resized = F.interpolate(batch_tensor, size=(TARGET_H, batch_tensor.shape[3]), mode='bilinear', align_corners=False)

        with torch.no_grad():
            preds, _, _ = model.forward(batch_resized)

        preds_post = postprocess(
            prediction=preds,
            num_classes=3,
            conf_thre=0.0001,
            nms_thre=0.3
        )

        pred_processed_all.extend(preds_post)

# -------------------------------
# Create Video with Bounding Boxes
# -------------------------------

visualize_event_tensor_video_with_predictions(h5_path, pred_processed_all, output_dir= os.path.join("visuals",sequence), max_frames=100, downsample=True)





if False:
    first_frame = 725
    final_frame = 821
    for frame_idx in range(first_frame, final_frame):  # Inclusive
    
        detections = pred_processed_all[frame_idx - first_frame]

        # Only draw boxes if detections exist
        if detections is not None:
            for obj in detections:
                x1, y1, x2, y2, obj_conf, class_conf, class_pred = obj
                w = x2 - x1
                h = y2 - y1
            
                



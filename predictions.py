import torch
from omegaconf import OmegaConf
from hydra import compose, initialize
from config.modifier import dynamically_modify_train_config
from modules.detection import Module
from models.detection.yolox.utils import postprocess  
import h5py
import numpy as np
import torch.nn.functional as F
import math
import cv2





# 1. Compose config using Hydra (like validation.py)
with initialize(config_path="config"):
    config = compose(config_name="val", overrides=["+experiment/gen4=default.yaml", 
                                                   "checkpoint=checkpoints/rvt-b-gen4.ckpt", 
                                                   "dataset=gen4",
                                                   "dataset.path=data/1mpx_proc",
                                                   "use_test_set=0",
                                                   "hardware.gpus=0",
                                                   "batch_size=1",
                                                   "hardware.num_workers=0",
                                                   "model.postprocess.confidence_threshold=0.001"])

dynamically_modify_train_config(config)
OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

# 2. Load your trained model checkpoint
ckpt_path = config.checkpoint
model = Module.load_from_checkpoint(ckpt_path, full_config=config)
model.eval()

model.to('cuda' if torch.cuda.is_available() else 'cpu')



h5_path = "data/1mpx_proc/scene1/event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations_ds2_nearest.h5"
h5_path_2 ="data/interlaken_00_b.h5"
# Open the h5 file
with h5py.File(h5_path, "r") as f:
    # List all available datasets
    print("Datasets:", list(f.keys()))
    
    # For example, use the first item (you may want to browse the keys for your setup)
    
    first_key = list(f.keys())[0]
    print("Loading sample:", first_key)
    
    
    ev_repr_np = f[first_key][()]  # read as numpy array


ev_repr_np = np.expand_dims(ev_repr_np, axis=0)

T = 10
# Take the first T frames for a single sample
ev_rep = ev_repr_np[[0][0]]  
sample = ev_rep 

print("Sample shape before rearranging:", sample.shape)  # (1198, 20, 360, 640)


# Convert to torch 

event_tensor = torch.cuda.FloatTensor(sample[:, :, :, :]) if torch.cuda.is_available() else torch.FloatTensor(sample[:, :, :, :]) 

if torch.cuda.is_available():
    event_tensor = event_tensor.to('cuda')
else:
    event_tensor = event_tensor.to('cpu')


print("Loaded event_tensor shape:", event_tensor.shape)
# Now you can pass this to your model
# predictions, losses, states = model.forward(event_tensor)

input_tensor = event_tensor[:5, :, :, :]  # Take the first 5 frames (20, 360, 640)

# ======================================================================
# === FIX: Resize the tensor height to 384 before the forward pass ===
# ======================================================================
TARGET_H = 384
current_w = input_tensor.shape[3] # Get the current width (640)

# Use interpolate for resizing. 'bilinear' is a good standard algorithm.
input_resized = F.interpolate(
    input_tensor,
    size=(TARGET_H, current_w), # Target shape: (H, W)
    mode='bilinear',
    align_corners=False
)


print("Input shape for model:", input_resized.shape)  # Should be (20, 360, 640)

# 4. Run prediction
with torch.no_grad():
    predictions, losses, states = model.forward(input_resized)

print(predictions[0][:][:])  # Check the shape of predictions

# torch to array
predictions_np = predictions[0].cpu().numpy()
print("Predictions shape:", predictions_np.shape)  # Should be (20, 360, 640)

print(predictions_np[0, :])  # Print a sample value


pred_processed = postprocess(
    prediction=predictions,
    num_classes=3,  # O el número de clases de tu modelo
    conf_thre= 0.0001,  # Umbral de confianza
    nms_thre=0.3  # Umbral NMS, si está en tu config
)

# elementos de la primera prediccion

""" x = pred_processed[0][0, 0]  # x1
y = pred_processed[0][0, 1]  # y1
w = pred_processed[0][0, 2] # width
h = pred_processed[0][0, 3]  # height
class_conf = pred_processed[0][0, 4]  # class confidence
print(f"x: {x}, y: {y}, w: {w}, h: {h}")  # Print the bounding box coordinates """

# CORREGIR A 
# Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)

for frame_idx, detections in enumerate(pred_processed):
    print(f"\n--- Frame {frame_idx} ---")
    if detections is None or len(detections) == 0:
        print("No detections.")
        continue

    for obj_idx, obj in enumerate(detections):
        x1,x2, y1, y2, obj_conf, class_conf, class_pred = obj

        w = x2 - x1
        h = y2 - y1

        if class_conf > 0.75:
            print(f"Object {obj_idx}: x1={x1:.2f}, y1={y1:.2f}, w={w:.2f}, h={h:.2f}, obj_conf={obj_conf:.2f}, class_conf={class_conf:.2f}, class_pred={int(class_pred)}")

        

import h5py
import numpy as np
import cv2
from utils.evaluation.prophesee.visualize.vis_utils import draw_bboxes
from data.utils.types import DataType  # Import DataType enum

# Load event data from .h5 file
h5_file = "data/dummy_gen1/test/17-04-04_11-00-13_cut_15_500000_60500000/event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations.h5"
with h5py.File(h5_file, "r") as f:
    # Inspect the structure of the HDF5 file
    def print_nested_keys(name, obj):
        print(name)
    f.visititems(print_nested_keys)

    # Access datasets within the file
    events = {
        DataType.EV_REPR: f["events/x"][:],
        "y": f["events/y"][:],
        "p": f["events/p"][:],  # Polarity
        "t": f["events/t"][:]   # Timestamps
    }

# Load annotations
annotations = np.load("data/dummy_gen1/test/17-04-04_11-00-13_cut_15_500000_60500000/event_representations_v2/stacked_histogram_dt=50_nbins=10/objframe_idx_2_repr_idx.npy")

# Visualize events and annotations
frame_img = np.zeros((240, 304, 3), dtype=np.uint8)  # Assuming Gen1 resolution
for frame_idx in range(len(annotations)):
    # Filter events for the current frame (if needed, based on timestamps)
    frame_events = {
        "x": events[DataType.EV_REPR],
        "y": events["y"],
        "p": events["p"]
    }
    frame_img[frame_events["y"], frame_events["x"]] = (255, 255, 255)  # White for events

    # Draw bounding boxes
    draw_bboxes(frame_img, annotations[frame_idx], labelmap=("car", "pedestrian"))

    # Save or display the frame
    cv2.imwrite(f"output/frame_{frame_idx}.png", frame_img)
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os
import h5py
from scripts.my_toolbox import *
import csv


SCALE_Y = 360 / 480  # = 0.75

split = "test"
root_dir = os.path.join("data/dsec_proc",split)
all_sequences = [item for item in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, item))]
confidence_thresholds = [0.5, 0.55, 0.60, 0.65 , 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
iou_thresholds = [0.5, 0.75]

# FOR TESTING PURPOSES ONLY
all_sequences = ["interlaken_00_b"] # DELETE BEFORE DOING AUTOMATIZATION

sequences_to_remove = ["interlaken_01_a"]

all_sequences = [item for item in all_sequences if item not in sequences_to_remove]


for sequence_name in tqdm(all_sequences):

    print(f"Processing {sequence_name}")

    num_frames = get_number_of_rgb_frames(sequence_name)

    # ----------------------- LOAD GT, RGB and EVENT PRED ------------------
    # ----------------------------------- PATHS ----------------------------
    # GT
    gt_pred_path = os.path.join("data/dsec_proc",split,sequence_name,"labels_v2","labels.npz")
    # RGB Predictions
    rgb_pred_path = os.path.join("rgb_pred",f"{sequence_name}.npy")
    # Event predictions
    event_pred_path = os.path.join("predictions",f"{sequence_name}.npz")

    # --------------------------- FILES ---------------------------
    gt_file = np.load(gt_pred_path, allow_pickle = True)

    # Load RGB predictions
    event_pred_file = np.load(event_pred_path, allow_pickle=True)

    # ------------------------ BY_FRAME VAR ---------------------
    rgb_preds_by_frame = get_rgb_by_frame(rgb_pred_path)

    event_preds_by_frame = convert_predictions(event_pred_file,num_frames)

    gt_by_frame = convert_gt(gt_file,num_frames)

    nms_events_by_frame = {}
    for frame_id, preds in event_preds_by_frame.items():
        nms_events_by_frame[frame_id] = apply_nms_to_frame(preds)

    # ----------------------- BBOXES --------------------------
    gt_bboxes = {}
    rgb_bboxes = {}
    event_bboxes = {}

    for idx in gt_by_frame:
        gt_bboxes[idx] = gt_by_frame[idx]

    for idx in rgb_preds_by_frame:
        rgb_bboxes[idx] = rgb_preds_by_frame[idx]

    for idx in nms_events_by_frame:
        event_bboxes[idx + 1] = nms_events_by_frame[idx]

    
    rgb_bboxes_filtered = filter_pred_by_confidence(rgb_bboxes, 0.1)
    event_bboxes_filtered = filter_pred_by_confidence(event_bboxes,0.7)

    print('------------------RGB & Event Detections----------------')
    
    #generate_metrics_csv(sequence_name,rgb_bboxes_filtered,event_bboxes_filtered,confidence_thresholds,gt_bboxes)

    process_drawn_images(sequence_name,gt_bboxes,rgb_bboxes_filtered,event_bboxes_filtered)

    create_video_from_frames(os.path.join('visuals/drawn',sequence_name),os.path.join('visuals/videos',f'{sequence_name}.mp4'))

    #---------------------- MERGE DETETCTIONS -----------------
    print('------------------MERGED DETETCTIONS----------------')

    merged_bboxes = nms_fusion(rgb_bboxes_filtered,event_bboxes_filtered)

    process_drawn_images(sequence_name,gt_bboxes,merged_bboxes,{})

    create_video_from_frames(os.path.join('visuals/drawn',sequence_name),os.path.join('visuals/videos',f'{sequence_name}_fused.mp4'))


    #-------------------- TRACKING -----------------------------------
    print('------------------TRACKING----------------')

    tracked_by_frame,tracker = tracking(merged_bboxes,sequence_name)

    process_drawn_tracked_images(sequence_name,tracked_by_frame,tracker)

    create_video_from_frames(os.path.join('visuals/tracked',sequence_name),os.path.join('visuals/videos',f'{sequence_name}_tracked.mp4'))












    





    

    

    

    



    



    
    
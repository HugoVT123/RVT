from collections import defaultdict
import numpy as np


def convert_predictions(predictions_file,num_frames):
    """
    Convert (frame_id, class_id, [x, y, w, h], confidence) predictions
    to {frame_id: [[x1, y1, x2, y2, confidence, class_id], ...]}
    """
    preds_by_frame = defaultdict(list)

    for pred in predictions_file['predictions']:
        
        x1,y1,x2,y2 = pred['bbox']

        # Define the original and new resolution
        original_resolution = 360
        new_resolution = 480

        # Calculate the scale factor
        scale_factor = new_resolution / original_resolution

        # Apply the scale factor to the coordinates
        y1_new =  y1 * scale_factor
        y2_new =  y2 * scale_factor

        frame_idx = pred['frame_idx']
        confidence = pred['confidence']

        if pred['class_id'] == 2:
            class_id = 1
        else:
            class_id = 0

        preds_by_frame[frame_idx].append([x1, y1_new, x2, y2_new, confidence, class_id])

    return preds_by_frame

def convert_gt(gt_file, num_frames):
    obj_2_idx_array = gt_file['objframe_idx_2_label_idx']
    gt_by_frame = defaultdict(list)

    count = 0
    for idx in obj_2_idx_array:

        label = gt_file['labels'][count]
        count = count + 1

        original_resolution = 360
        new_resolution = 480

        # Calculate the scale factor
        scale_factor = new_resolution / original_resolution

        x, y, w, h = label['x'], label['y'], label['w'], label['h']
        x1 = x
        y1 = y 
        x2 = x + w
        y2 = y1 + h

        y1_new =  y1 * scale_factor
        y2_new =  y2 * scale_factor

        confidence = label['class_confidence']
        if label['class_id'] == 2:
            class_id = 1
        else:
            class_id = 0
        track_id = label['track_id']

        gt_by_frame[idx].append([x1, y1_new, x2, y2_new, confidence, class_id,track_id])

    return gt_by_frame

def get_rgb_by_frame(rgb_pred_path):

    # Load RGB predictions
    rgb_pred_file = np.load(rgb_pred_path, allow_pickle=True)

    rgb_by_frame = defaultdict(list)

    for item in rgb_pred_file:
        x1,y1,x2,y2 = item['bbox']

        frame_idx = item['frame_idx']
        confidence = item['confidence']
        rgb_by_frame[item['frame_idx']]

        if item['class_id'] == 3:
            class_id = 1

        elif item['class_id'] == 1:
            class_id = 0

        else:
            continue

        rgb_by_frame[frame_idx].append([x1, y1, x2, y2, confidence, class_id])
    
    return rgb_by_frame

def change_class_for_metrics(arr):
    for sublist in arr:
        if sublist and sublist[-1] == 2:
            sublist[-1] = 1
    return arr


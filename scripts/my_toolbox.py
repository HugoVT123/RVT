from collections import defaultdict
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from mean_average_precision import MetricBuilder
import csv
import torchvision.ops as ops  # for NMS
import torch
import h5py
import os
from pathlib import Path
from ensemble_boxes import weighted_boxes_fusion
from deep_sort_realtime.deepsort_tracker import DeepSort
import motmetrics as mm
from ultralytics.trackers.byte_tracker import BYTETracker
from argparse import Namespace
import supervision as sv
from supervision.detection.core import Detections
from PIL import Image
import shutil

def apply_nms_to_frame(predictions, iou_threshold=0.5):
    boxes = []
    scores = []
    class_ids = []
    
    for box in predictions:
        boxes.append(box[:4])
        scores.append(box[4])
        class_ids.append(int(box[5]))  # Ensure class ID is integer
    
    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)
    class_ids = np.array(class_ids)

    # Perform NMS per class
    final_indices = []
    for cls in np.unique(class_ids):
        cls_mask = (class_ids == cls)
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        
        if cls_boxes.shape[0] == 0:
            continue
        
        keep = ops.nms(cls_boxes, cls_scores, iou_threshold)
        cls_indices = np.where(cls_mask)[0][keep.numpy()]
        final_indices.extend(cls_indices)

    return [predictions[i] for i in final_indices]

def get_number_of_rgb_frames(sequence):
    switcher = {
        'interlaken_00_a': 1143,
        'interlaken_00_b': 1617, 
        'interlaken_01_a': 2263, 
        'thun_01_a': 197, 
        'thun_01_b': 1079, 
        'thun_02_a': 3901, 
        'zurich_city_12_a': 765, 
        'zurich_city_13_a': 379, 
        'zurich_city_13_b': 315, 
        'zurich_city_14_a': 439, 
        'zurich_city_14_b': 577, 
        'zurich_city_14_c': 1191, 
        'zurich_city_15_a': 1239,
        'zurich_city_00_b': 1463,
        'interlaken_00_d': 1991
    }
    # Return the corresponding integer for the string, or a default value if the string is not found
    # If the sequence is not found, return 0 or handle it as needed
    if sequence not in switcher:
        print(f"Warning: Sequence '{sequence}' not found in switcher. Returning default value of 0.")
    return switcher.get(sequence, 0)  # Default is 0 if the case is not found

def evaluate_precision_recall(gt_list, pred_list, iou_thresh=0.5):
    """
    gt_list: dict of frame_idx -> list of arrays [x1, y1, x2, y2, conf, class] (GT conf is ignored)
    pred_list: dict of frame_idx -> list of arrays [x1, y1, x2, y2, conf, class]
    iou_thresh: IoU threshold to consider a prediction a TP
    """
    def compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    gt = defaultdict(list)
    preds_by_class = defaultdict(list)

    # Step 1: Aggregate all boxes across frames by class
    for frame_idx in gt_list:
        if frame_idx not in pred_list:
            continue

        gt_bboxes_per_frame = gt_list[frame_idx]
        pred_bboxes_per_frame = pred_list[frame_idx]

        for g in gt_bboxes_per_frame:
            box = g[:4]
            cls = int(g[5])
            gt[cls].append(box)

        for p in pred_bboxes_per_frame:
            box = p[:4]
            conf = p[4]
            cls = int(p[5])
            preds_by_class[cls].append((box, conf))

    results = {}
    all_classes = sorted(set(gt.keys()) | set(preds_by_class.keys()))

    # Step 2: Match predictions to GTs and compute PR per class
    for cls in all_classes:
        gt_boxes = gt.get(cls, [])
        preds = preds_by_class.get(cls, [])

        preds.sort(key=lambda x: x[1], reverse=True)

        y_true = []
        y_scores = []
        used_gt = set()

        for pred_box, conf in tqdm(preds, desc=f"Processing class {cls}"):
            best_iou = 0
            best_idx = -1

            for idx, gt_box in enumerate(gt_boxes):
                if idx in used_gt:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= iou_thresh:
                y_true.append(1)
                used_gt.add(best_idx)
            else:
                y_true.append(0)
            y_scores.append(conf)

        # Add false negatives
        num_fn = len(gt_boxes) - len(used_gt)
        if num_fn > 0:
            y_true.extend([1] * num_fn)
            y_scores.extend([0.0] * num_fn)

        if len(set(y_true)) < 2:
            print(f"Skipping class {cls} (not enough pos/neg examples)")
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)

        results[cls] = {
            'precision': precision,
            'recall': recall,
            'ap': ap
        }

    # Step 3: Plot PR curves
    for cls, data in results.items():
        plt.plot(data['recall'], data['precision'], label=f'Class {cls} AP = {data["ap"]:.2f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves by Class")
    plt.legend()
    plt.grid(True)
    plt.show()

    return results

def evaluate_precision_recall(gt_list, pred_list, iou_thresh=0.5):
    """
    gt_list: dict of frame_idx -> list of arrays [x1, y1, x2, y2, conf, class] (GT conf is ignored)
    pred_list: dict of frame_idx -> list of arrays [x1, y1, x2, y2, conf, class]
    iou_thresh: IoU threshold to consider a prediction a TP
    """
    def compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    gt = defaultdict(list)
    preds_by_class = defaultdict(list)

    # Step 1: Aggregate all boxes across frames by class
    for frame_idx in gt_list:
        if frame_idx not in pred_list:
            continue

        gt_bboxes_per_frame = gt_list[frame_idx]
        pred_bboxes_per_frame = pred_list[frame_idx]

        for g in gt_bboxes_per_frame:
            box = g[:4]
            cls = int(g[5])
            gt[cls].append(box)

        for p in pred_bboxes_per_frame:
            box = p[:4]
            conf = p[4]
            cls = int(p[5])
            preds_by_class[cls].append((box, conf))

    results = {}
    all_classes = sorted(set(gt.keys()) | set(preds_by_class.keys()))

    # Step 2: Match predictions to GTs and compute PR per class
    for cls in all_classes:
        gt_boxes = gt.get(cls, [])
        preds = preds_by_class.get(cls, [])

        preds.sort(key=lambda x: x[1], reverse=True)

        y_true = []
        y_scores = []
        used_gt = set()

        for pred_box, conf in tqdm(preds, desc=f"Processing class {cls}"):
            best_iou = 0
            best_idx = -1

            for idx, gt_box in enumerate(gt_boxes):
                if idx in used_gt:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= iou_thresh:
                y_true.append(1)
                used_gt.add(best_idx)
            else:
                y_true.append(0)
            y_scores.append(conf)

        # Add false negatives
        num_fn = len(gt_boxes) - len(used_gt)
        if num_fn > 0:
            y_true.extend([1] * num_fn)
            y_scores.extend([0.0] * num_fn)

        if len(set(y_true)) < 2:
            print(f"Skipping class {cls} (not enough pos/neg examples)")
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)

        results[cls] = {
            'precision': precision,
            'recall': recall,
            'ap': ap
        }

    # Step 3: Plot PR curves
    for cls, data in results.items():
        plt.plot(data['recall'], data['precision'], label=f'Class {cls} AP = {data["ap"]:.2f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves by Class")
    plt.legend()
    plt.grid(True)
    plt.show()

    return results

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

def draw_bboxes_on_image(gt_boxes, pred_boxes, idx, image_size=(480, 640), confidence=0.5,sequence_name="0"):
    height, width = image_size
    image = cv2.imread(f"{sequence_name}/0000{idx}.png")
    if image is None:
        raise ValueError(f"Image  could not be loaded.")

    # Draw GT boxes in green
    for box in gt_boxes:
        x1, y1, x2, y2, _, cls = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"GT:{cls}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw predicted boxes in red
    for box in pred_boxes:
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        if conf > confidence:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, f"P:{int(cls)} {conf:.2f}", (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Show the image
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Green = Ground Truth, Red = Prediction")
    plt.show()

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

def get_metrics(preds_filtered, gt_bboxes, original_classes = [0, 1],iou_thresholds=[0.5, 0.75]):

    class_id_map = {orig: idx for idx, orig in enumerate(sorted(original_classes))}

    # Initialize metric
    num_classes = len(original_classes)
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=num_classes)

    # Loop over frames
    for frame_id in gt_bboxes.keys():
        gt_raw = gt_bboxes.get(frame_id, [])
        pred_raw = preds_filtered.get(frame_id, [])

        # Convert GT

        gt = np.array([
            [x[0], x[1], x[2], x[3], class_id_map[int(x[5])], 0, 0]  # class_id is at index 5
            for x in gt_raw if int(x[5]) in class_id_map
        ], dtype=np.float32)

        # Convert Predictions
        preds = np.array([
            [x[0], x[1], x[2], x[3], class_id_map[int(x[5])], x[4]]  # class_id is at index 5, confidence is at index 4
            for x in pred_raw if int(x[5]) in class_id_map
        ], dtype=np.float32)


        metric_fn.add(preds, gt)

    # Compute and print final metrics
    metrics = metric_fn.value(iou_thresholds)

    return metrics

def filter_pred_by_confidence(preds_by_frame, conf_thresholds):
    """
    conf_thresholds: dict like {0: 0.5, 1: 0.6}
    """
    for cls, thresh in conf_thresholds.items():
        if not (0 <= thresh <= 1):
            raise ValueError(f"Confidence threshold for class {cls} must be between 0 and 1")

    preds_filtered = {
        frame: [det for det in dets if det[5] in conf_thresholds and det[4] >= conf_thresholds[det[5]]]
        for frame, dets in preds_by_frame.items()
    }

    return preds_filtered


def generate_metrics_csv(sequence_name, rgb_preds_by_frame, event_preds_by_frame, confidence_thresholds, gt_bboxes):

    def write_metrics_file(filename, preds_by_frame):
        with open(filename, mode='w', newline='') as file:
            csv_writer = csv.writer(file, delimiter=';')
            csv_writer.writerow(['conf', 'class', 'mAP', 'mAP50', 'P50', 'R50','mAP75', 'P75', 'R75'])

            for conf in confidence_thresholds:
                preds_filtered = filter_pred_by_confidence(preds_by_frame, {0: conf, 1: conf})
                metrics = get_metrics(preds_filtered, gt_bboxes)

                mAP = metrics['mAP']
                

                # Check for necessary keys
                if 0.5 not in metrics or 0.75 not in metrics:
                    print(f"Missing IoU thresholds at conf={conf}. Keys: {metrics.keys()}")
                    continue

                for class_id in [0, 1]:
                    if class_id not in metrics[0.5] or class_id not in metrics[0.75]:
                        print(f"Missing class {class_id} at conf={conf}")
                        continue

                    try:
                        results_50 = metrics[0.5][class_id]
                        results_75 = metrics[0.75][class_id]

                        mAP50 = np.average(results_50['ap'])
                        p50 = np.average(results_50['precision'])
                        r50 = np.average(results_50['recall'])

                        mAP75 = np.average(results_75['ap'])
                        p75 = np.average(results_75['precision'])
                        r75 = np.average(results_75['recall'])

                        csv_writer.writerow([conf, class_id, mAP, mAP50, p50, r50, mAP75, p75, r75])
                    except Exception as e:
                        print(f"Error processing class {class_id} at conf={conf}: {e}")

    # Generate both RGB and Event metrics files
    write_metrics_file(f'metrics/{sequence_name}_rgb.csv', rgb_preds_by_frame)
    write_metrics_file(f'metrics/{sequence_name}_event.csv', event_preds_by_frame)

def draw_bboxes_on_image_v2(idx,img_path, gt_bboxes, rgb_bboxes, event_bboxes,hybrid_bboxes,tracked_by_frame, confidence=0.5,frame_weight=0.75,event_weight=0.25):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Image could not be loaded: {img_path}")
    
    # this is to get the sequence_name
    image_path = Path(img_path)
    sequence_name = image_path.parts[2]
    
    
    #image = get_event_frame_rgb(idx,image,sequence_name=sequence_name,frame_weight=frame_weight,event_weight=event_weight)
    image = get_event_frame_rgb_dominant(idx,image,sequence_name=sequence_name,frame_weight=frame_weight,event_weight=event_weight)

    if event_weight > 0:
        image = adjust_brightness_add(image,25)

    # Colors BGR

    rgb_color = (0, 128, 255)  # Orange
    event_color = (0, 255, 0)  # Green
    hybrid_color = (255, 0, 247)  # Pink
    tracked_color = (255, 255, 0)  # Cyan

    gt_color_filling = (0, 255, 255)  # White filling for GT boxes
    gt_color_border = gt_color_filling
    #gt_color_border = (255, 255, 255)  # Black border for GT boxes


    # Draw GT boxes with white fill and black border
    if idx in gt_bboxes:
        for det in gt_bboxes[idx]:
            x1, y1, x2, y2, _, cls, track_id = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # ensure ints

            # Draw outer black border
            cv2.rectangle(image, (x1, y1), (x2, y2), gt_color_border, 3)
            # Draw inner white box
            cv2.rectangle(image, (x1, y1), (x2, y2), gt_color_filling, 1)

            # Draw text with black outline then white fill for readability
            label = f"GT:{cls} ID:{track_id}"
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color_border, 2)  # black outline
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color_filling, 1)  # white fill

    



    # Draw predicted RGB boxes in ORANGE
    if idx in rgb_bboxes:
        for det in rgb_bboxes[idx]:
            x1, y1, x2, y2, conf, cls = det
            if conf > confidence:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                cv2.rectangle(image, (x1, y1), (x2, y2), rgb_color, 2)
                cv2.putText(image, f"P:{int(cls)} {conf:.2f}", (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb_color, 2)
                
    # Draw predicted EVENT boxes in GREEN
    if idx in event_bboxes:
        for det in event_bboxes[idx]:
            x1, y1, x2, y2, conf, cls = det
            if conf > confidence:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(image, (x1, y1), (x2, y2), event_color, 2)
                cv2.putText(image, f"P:{int(cls)} {conf:.2f}", (x1, y2 + 15 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, event_color, 2)
                
    # Draw predicted HYBRID boxes in PINK            
    if idx in hybrid_bboxes:
        for det in hybrid_bboxes[idx]:
            x1, y1, x2, y2, conf, cls = det
            if conf > confidence:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                cv2.rectangle(image, (x1, y1), (x2, y2), hybrid_color, 2)
                cv2.putText(image, f"P:{int(cls)} {conf:.2f}", (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, hybrid_color, 2)
                
    tracks = tracked_by_frame.get(idx, [])

    for det in tracks:
        x1, y1, x2, y2, track_id, cls = det

        cv2.rectangle(image, (x1, y1), (x2, y2), tracked_color, 2)
        cv2.putText(image, f'ID {track_id} | Cls {cls}', (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, tracked_color, 2)
                

                
    return image

def draw_bboxes_on_image(idx,img_path, gt_bboxes, rgb_bboxes, event_bboxes, confidence=0.5,frame_weight=0.75,event_weight=0.25):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Image could not be loaded: {img_path}")
    
    # this is to get the sequence_name
    image_path = Path(img_path)
    sequence_name = image_path.parts[2]
    
    
    #image = get_event_frame_rgb(idx,image,sequence_name=sequence_name,frame_weight=frame_weight,event_weight=event_weight)
    image = get_event_frame_rgb_dominant(idx,image,sequence_name=sequence_name,frame_weight=frame_weight,event_weight=event_weight)

    # Draw GT boxes in blue
    if idx in gt_bboxes:
        for det in gt_bboxes[idx]:
            x1, y1, x2, y2, _, cls, track_id = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # asegurar ints
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, f"Track:{track_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw predicted RGB boxes in orange
    
    if idx in rgb_bboxes:
        for det in rgb_bboxes[idx]:
            x1, y1, x2, y2, conf, cls = det
            if conf > confidence:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 247), 2)
                cv2.putText(image, f"P:{int(cls)} {conf:.2f}", (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 247), 2)

                """ cv2.rectangle(image, (x1, y1), (x2, y2), (0, 182, 255), 2)
                cv2.putText(image, f"P:{int(cls)} {conf:.2f}", (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 182, 255), 2) """

    # Draw predicted EVENT boxes in green
    
    if idx in event_bboxes:
        for det in event_bboxes[idx]:
            x1, y1, x2, y2, conf, cls = det
            if conf > confidence:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"P:{int(cls)} {conf:.2f}", (x1, y2 + 15 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

def get_event_frame_rgb(index, frame_img, sequence_name, downsample=True, split='test', frame_weight=1.0, event_weight=0.5):
    """
    Retrieves and blends an event frame with an RGB image, allowing independent control
    over the visibility of both images.

    Args:
        index (int): The index of the event frame to retrieve.
        frame_img (numpy.ndarray): The RGB image (background) to blend with.
                                   Expected shape (H, W, 3) and dtype np.uint8.
        sequence_name (str): The name of the sequence (e.g., 'sequence_01').
        downsample (bool, optional): Whether to downsample the event frame
                                     to match frame_img dimensions. Defaults to True.
        split (str, optional): The dataset split ('test', 'train', 'val'). Defaults to 'test'.
        frame_weight (float, optional): The weight given to the original frame_img.
                                        Value typically between 0.0 (transparent) and 1.0 (fully visible).
                                        Can be higher for emphasis. Defaults to 1.0.
        event_weight (float, optional): The weight given to the event_rgb image.
                                        Value typically between 0.0 (transparent) and 1.0 (fully visible).
                                        Can be higher for emphasis. Defaults to 0.5.

    Returns:
        numpy.ndarray: The blended image (RGB). If both weights are 0, or very low,
                       the result will approach a black image (due to how addWeighted works
                       with zero inputs and gamma=0). To achieve a white background,
                       we'll explicitly blend with white if both are zero.
    """

    h5_path = os.path.join('data/dsec_proc', split, sequence_name,
                           "event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations_ds2_nearest.h5")

    try:
        with h5py.File(h5_path, 'r') as f:
            data = f['/data']  # Shape: (N, 2*T, H, W)

            if index >= data.shape[0]:
                # Adjust index if it's out of bounds, or handle as per your needs
                index = data.shape[0] - 1
                if index < 0:
                    print(f"Warning: Dataset for {sequence_name} is empty. Returning white image.")
                    return np.full(frame_img.shape, 255, dtype=np.uint8) # Return white if no data
                print(f"Warning: Index {index+1} out of bounds. Using last available frame at index {index}.")

            frame = data[index]  # Shape: (2*T, H, W)
            T = frame.shape[0] // 2
            pos = np.sum(frame[:T], axis=0)
            neg = np.sum(frame[T:], axis=0)

            if downsample:
                target_size = (frame_img.shape[1], frame_img.shape[0])
                pos = cv2.resize(pos, target_size, interpolation=cv2.INTER_NEAREST)
                neg = cv2.resize(neg, target_size, interpolation=cv2.INTER_NEAREST)

            # Normalize to 0–255
            pos_norm = cv2.normalize(pos, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            neg_norm = cv2.normalize(neg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Create RGB event image: Red = positive, Blue = negative
            event_rgb = np.zeros_like(frame_img, dtype=np.uint8)
            event_rgb[..., 2] = pos_norm  # Red channel
            event_rgb[..., 0] = neg_norm  # Blue channel

            # Special case: if both weights are 0 or very close to 0, return a white image.
            # Otherwise, use addWeighted.
            # Using a small epsilon to catch near-zero weights
            epsilon = 1e-6
            if abs(frame_weight) < epsilon and abs(event_weight) < epsilon:
                return np.full(frame_img.shape, 255, dtype=np.uint8) # Return white image

            # Now, blend based on the independent weights
            # The result of addWeighted with gamma=0 is (src1*alpha + src2*beta).
            # If both alpha and beta are 0, the result is 0 (black).
            # If we want a white background when both are low, we need to handle it.
            # A common approach for independent control is:
            # 1. Start with a black image (or white if that's the default background).
            # 2. Add the frame_img multiplied by its weight.
            # 3. Add the event_rgb multiplied by its weight.

            # We can use addWeighted in two steps, or create a background first.
            # Let's create a base image to blend onto. If frame_weight and event_weight are low,
            # this base will be more visible.
            # Since the original requirement was 'if rgb one disappears background should be white',
            # and now 'control how much of each', if *both* are very low, you want white.
            # Let's ensure a white base if both primary images are minimal.

            # Start with a black canvas
            blended_img = np.zeros_like(frame_img, dtype=np.float32) # Use float for intermediate calculations

            # Add the weighted frame image
            blended_img += frame_img.astype(np.float32) * frame_weight

            # Add the weighted event image
            blended_img += event_rgb.astype(np.float32) * event_weight

            # Clip values to 0-255 and convert back to uint8
            blended_img = np.clip(blended_img, 0, 255).astype(np.uint8)

            return blended_img

    except FileNotFoundError:
        print(f"Error: H5 file not found at {h5_path}. Returning white image.")
        return np.full(frame_img.shape, 255, dtype=np.uint8)
    except Exception as e:
        print(f"An error occurred: {e}. Returning white image.")
        return np.full(frame_img.shape, 255, dtype=np.uint8)

def get_event_frame_rgb_binary(index, frame_img, sequence_name, downsample=True, split='test',
                               frame_weight=1.0, event_weight=0.5):
    """
    Retrieves and blends a binary event frame with an RGB image.
    Each pixel with any positive polarity becomes fully red (255,0,0),
    each with negative polarity becomes fully blue (0,0,255).
    No intensity scaling – pure binary event visualization.

    Args:
        index (int): Index of the event frame to retrieve.
        frame_img (np.ndarray): RGB image of shape (H, W, 3), dtype=np.uint8.
        sequence_name (str): Sequence folder name.
        downsample (bool): Whether to resize the event data to match the frame. Default is True.
        split (str): Dataset split ('train', 'val', 'test').
        frame_weight (float): Visibility of the frame image.
        event_weight (float): Visibility of the event image.

    Returns:
        np.ndarray: RGB image with blended original and binary red/blue event overlay.
    """
    h5_path = os.path.join('data/dsec_proc', split, sequence_name,
                           "event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations_ds2_nearest.h5")

    try:
        with h5py.File(h5_path, 'r') as f:
            data = f['/data']

            if index >= data.shape[0]:
                index = data.shape[0] - 1
                if index < 0:
                    print(f"Warning: Dataset for {sequence_name} is empty. Returning white image.")
                    return np.full(frame_img.shape, 255, dtype=np.uint8)

                print(f"Warning: Index {index+1} out of bounds. Using last available frame at index {index}.")

            frame = data[index]  # Shape: (2*T, H, W)
            T = frame.shape[0] // 2
            pos = np.sum(frame[:T], axis=0)
            neg = np.sum(frame[T:], axis=0)

            if downsample:
                target_size = (frame_img.shape[1], frame_img.shape[0])
                pos = cv2.resize(pos, target_size, interpolation=cv2.INTER_NEAREST)
                neg = cv2.resize(neg, target_size, interpolation=cv2.INTER_NEAREST)

            # Create binary masks where events occurred
            pos_mask = pos > 0
            neg_mask = neg > 0

            # Create binary RGB image: Red = pos, Blue = neg
            event_rgb = np.zeros_like(frame_img, dtype=np.uint8)
            event_rgb[..., 2][pos_mask] = 255  # Red channel
            event_rgb[..., 0][neg_mask] = 255  # Blue channel

            epsilon = 1e-6
            if abs(frame_weight) < epsilon and abs(event_weight) < epsilon:
                return np.full(frame_img.shape, 255, dtype=np.uint8)

            # Blend the binary event image with the original frame
            blended_img = np.zeros_like(frame_img, dtype=np.float32)
            blended_img += frame_img.astype(np.float32) * frame_weight
            blended_img += event_rgb.astype(np.float32) * event_weight
            blended_img = np.clip(blended_img, 0, 255).astype(np.uint8)

            return blended_img

    except FileNotFoundError:
        print(f"Error: H5 file not found at {h5_path}. Returning white image.")
        return np.full(frame_img.shape, 255, dtype=np.uint8)
    except Exception as e:
        print(f"An error occurred: {e}. Returning white image.")
        return np.full(frame_img.shape, 255, dtype=np.uint8)

def get_event_frame_rgb_dominant(index, frame_img, sequence_name, downsample=True, split='test',
                                        frame_weight=1.0, event_weight=0.5):
    """
    Retrieves and blends an event frame with an RGB image.
    Each pixel is fully red or blue based on polarity dominance.
    No normalization; pure binary dominance coloring.

    Args:
        index (int): Index of the event frame.
        frame_img (np.ndarray): RGB image (background).
        sequence_name (str): Dataset sequence name.
        downsample (bool): Resize event data to match frame_img.
        split (str): Dataset split.
        frame_weight (float): Weight for the RGB frame.
        event_weight (float): Weight for the event overlay.

    Returns:
        np.ndarray: Blended RGB image with red/blue binary overlay.
    """
    h5_path = os.path.join('data/dsec_proc', split, sequence_name,
                           "event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations_ds2_nearest.h5")

    try:
        with h5py.File(h5_path, 'r') as f:
            data = f['/data']

            if index >= data.shape[0]:
                index = data.shape[0] - 1
                if index < 0:
                    print(f"Warning: Empty dataset for {sequence_name}. Returning white image.")
                    return np.full(frame_img.shape, 255, dtype=np.uint8)

                print(f"Warning: Index {index+1} out of bounds. Using last available frame at index {index}.")

            frame = data[index]
            T = frame.shape[0] // 2
            pos = np.sum(frame[:T], axis=0)
            neg = np.sum(frame[T:], axis=0)

            if downsample:
                target_size = (frame_img.shape[1], frame_img.shape[0])
                pos = cv2.resize(pos, target_size, interpolation=cv2.INTER_NEAREST)
                neg = cv2.resize(neg, target_size, interpolation=cv2.INTER_NEAREST)

            # Determine dominant polarity
            red_mask = (pos > neg)
            blue_mask = (neg > pos)

            # Create RGB binary event image
            event_rgb = np.zeros_like(frame_img, dtype=np.uint8)
            event_rgb[..., 2][red_mask] = 255  # Red channel full
            event_rgb[..., 0][blue_mask] = 255  # Blue channel full

            # Handle zero weights (transparent)
            epsilon = 1e-6
            if abs(frame_weight) < epsilon and abs(event_weight) < epsilon:
                return np.full(frame_img.shape, 255, dtype=np.uint8)

            # Blend images
            blended_img = np.zeros_like(frame_img, dtype=np.float32)
            blended_img += frame_img.astype(np.float32) * frame_weight
            blended_img += event_rgb.astype(np.float32) * event_weight
            blended_img = np.clip(blended_img, 0, 255).astype(np.uint8)

            return blended_img

    except FileNotFoundError:
        print(f"Error: H5 file not found at {h5_path}. Returning white image.")
        return np.full(frame_img.shape, 255, dtype=np.uint8)
    except Exception as e:
        print(f"An error occurred: {e}. Returning white image.")
        return np.full(frame_img.shape, 255, dtype=np.uint8)

def get_event_frame_rgb_dominant_v2(index, frame_img, sequence_name, downsample=True, split='test',
                                 frame_weight=1.0, event_weight=0.5):
    """
    Retrieves and blends an event frame with an RGB image.
    Each pixel is fully red or blue based on polarity dominance.
    No normalization; pure binary dominance coloring.

    Args:
        index (int): Index of the event frame.
        frame_img (np.ndarray): RGB image (background).
        sequence_name (str): Dataset sequence name.
        downsample (bool): Resize event data to match frame_img.
        split (str): Dataset split.
        frame_weight (float): Weight for the RGB frame.
        event_weight (float): Weight for the event overlay.

    Returns:
        np.ndarray: Blended RGB image with red/blue binary overlay.
    """
    h5_path = os.path.join('data/dsec_proc', split, sequence_name,
                           "event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations_ds2_nearest.h5")

    try:
        with h5py.File(h5_path, 'r') as f:
            data = f['/data']

            if index >= data.shape[0]:
                index = data.shape[0] - 1
                if index < 0:
                    print(f"Warning: Empty dataset for {sequence_name}. Returning white image.")
                    return np.full(frame_img.shape, 255, dtype=np.uint8)

                print(f"Warning: Index {index+1} out of bounds. Using last available frame at index {index}.")

            frame = data[index]
            T = frame.shape[0] // 2
            pos = np.sum(frame[:T], axis=0)
            neg = np.sum(frame[T:], axis=0)

            if downsample:
                target_size = (frame_img.shape[1], frame_img.shape[0])
                pos = cv2.resize(pos, target_size, interpolation=cv2.INTER_NEAREST)
                neg = cv2.resize(neg, target_size, interpolation=cv2.INTER_NEAREST)

            # Determine dominant polarity
            red_mask = (pos > neg)
            blue_mask = (neg > pos)
            # Find pixels where there are events (either positive or negative)
            event_pixels_mask = (pos > 0) | (neg > 0)

            # Create RGB binary event image
            event_rgb = np.zeros_like(frame_img, dtype=np.uint8)
            event_rgb[..., 2][red_mask] = 255  # Red channel full (BGR format: Blue, Green, Red)
            event_rgb[..., 0][blue_mask] = 255  # Blue channel full

            epsilon = 1e-6
            if abs(frame_weight) < epsilon:
                # If frame_weight is zero, initialize with white background
                blended_img = np.full(frame_img.shape, 255, dtype=np.float32)

                # Now, blend the event_rgb onto this white background
                # This ensures event colors are visible, and non-event areas remain white.
                # Only apply event color where events exist.
                # For event pixels: blend event_rgb with white (255) based on event_weight
                # For non-event pixels: keep them white (255)
                
                # Create a temporary image for blending events onto white
                events_on_white = np.full(frame_img.shape, 255, dtype=np.float32)
                # Overlay the event colors onto this temporary white image
                events_on_white[red_mask, :] = [0, 0, 255] # Red event pixels (BGR)
                events_on_white[blue_mask, :] = [255, 0, 0] # Blue event pixels (BGR)
                
                # Now blend the original white background with the event_rgb_on_white
                # The alpha for event_rgb is `event_weight`. The alpha for the background (white) is `1 - event_weight`.
                # If event_weight is 0, it should be pure white. If 1, pure events.
                
                # A simple way for a binary event overlay on white:
                blended_img = np.full(frame_img.shape, 255, dtype=np.float32) # Start with full white
                
                # Apply event colors directly only where events are present
                # Use a masked assignment with weighted colors
                # For red events
                blended_img[..., 2][red_mask] = (event_rgb[..., 2][red_mask].astype(np.float32) * event_weight) + (255 * (1 - event_weight))
                blended_img[..., 1][red_mask] = (event_rgb[..., 1][red_mask].astype(np.float32) * event_weight) + (255 * (1 - event_weight))
                blended_img[..., 0][red_mask] = (event_rgb[..., 0][red_mask].astype(np.float32) * event_weight) + (255 * (1 - event_weight))

                # For blue events
                blended_img[..., 2][blue_mask] = (event_rgb[..., 2][blue_mask].astype(np.float32) * event_weight) + (255 * (1 - event_weight))
                blended_img[..., 1][blue_mask] = (event_rgb[..., 1][blue_mask].astype(np.float32) * event_weight) + (255 * (1 - event_weight))
                blended_img[..., 0][blue_mask] = (event_rgb[..., 0][blue_mask].astype(np.float32) * event_weight) + (255 * (1 - event_weight))
                
                # This direct assignment is cleaner for binary events on white
                # Initialize with white
                blended_img = np.full(frame_img.shape, 255, dtype=np.float32)
                
                # Set event pixels
                blended_img[red_mask, 2] = 255 * event_weight + 255 * (1 - event_weight) # Red for red_mask
                blended_img[red_mask, 1] = 0   * event_weight + 255 * (1 - event_weight)
                blended_img[red_mask, 0] = 0   * event_weight + 255 * (1 - event_weight)

                blended_img[blue_mask, 0] = 255 * event_weight + 255 * (1 - event_weight) # Blue for blue_mask
                blended_img[blue_mask, 1] = 0   * event_weight + 255 * (1 - event_weight)
                blended_img[blue_mask, 2] = 0   * event_weight + 255 * (1 - event_weight)

                # The problem with the above is that it doesn't correctly handle the non-event parts of the event_rgb.
                # A more general approach is needed.
                
                # Simplest working solution:
                # 1. Start with a white image
                blended_img = np.full(frame_img.shape, 255, dtype=np.float32)
                
                # 2. Where there are events, blend the event color with the white background
                # This is like alpha blending (source_color * alpha + background_color * (1-alpha))
                # For event pixels, the "source_color" is event_rgb, "background_color" is white (255)
                # For non-event pixels, it remains white (255)
                
                # Iterate over channels to apply blending only to event pixels
                for c in range(3): # B, G, R
                    # If event_rgb has a non-zero value for this channel (meaning it's an event pixel that contributes to this color)
                    # OR if it's an event pixel but that channel is zero (e.g., green channel for pure red)
                    # we want to blend the event color.
                    # Otherwise, keep it white.

                    # Identify pixels that are part of an event and need color modification
                    # This applies to all channels for a given event pixel
                    channel_event_pixels = event_rgb[:, :, c] > 0
                    
                    # Apply the blend only where event_pixels_mask is True
                    # If event_rgb is 0 for a channel, but it's an event, we blend 0 with 255
                    blended_img[:, :, c] = np.where(event_pixels_mask,
                                                    (event_rgb[:, :, c].astype(np.float32) * event_weight) + (255 * (1 - event_weight)),
                                                    blended_img[:, :, c]) # Keep white for non-event pixels
                
            else:
                # Original blending logic when frame_weight is not zero
                blended_img = np.zeros_like(frame_img, dtype=np.float32)
                blended_img += frame_img.astype(np.float32) * frame_weight
                blended_img += event_rgb.astype(np.float32) * event_weight

            blended_img = np.clip(blended_img, 0, 255).astype(np.uint8)

            return blended_img

    except FileNotFoundError:
        print(f"Error: H5 file not found at {h5_path}. Returning white image.")
        return np.full(frame_img.shape, 255, dtype=np.uint8)
    except Exception as e:
        print(f"An error occurred: {e}. Returning white image.")
        return np.full(frame_img.shape, 255, dtype=np.uint8)

def create_video_from_frames(input_folder, output_video_path, fps=20):
    images = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not images:
        print("No image files found.")
        return

    first_frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, _ = first_frame.shape
    size = (width, height)

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for img_name in tqdm(images):
        img_path = os.path.join(input_folder, img_name)
        frame = cv2.imread(img_path)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, size)
        out.write(frame)

    out.release()
    print(f"Video saved to {output_video_path}")

def process_drawn_images(sequence_name,gt_bboxes,rgb_bboxes_filtered,event_bboxes_filtered,weights=[0.5,0.5]):
        # Define the folder containing your images
    image_folder = os.path.join('data/rgb',sequence_name,'images/left/distorted') 

    # List all files in the directory
    all_files = os.listdir(image_folder)

    # Filter for image files (you might need to expand this list)
    image_extensions = ('.png')
    image_files = sorted([f for f in all_files if f.lower().endswith(image_extensions)])


    # Loop through the sorted image files
    for i, filename in tqdm(enumerate(image_files[:-1]), total=len(image_files) - 1):
        full_path = os.path.join(image_folder, filename)

        # THIS FUNCTIONS HAS THE PARAMETERS FOR THE WEIGHTS OF THE ALPHAS OF THE IMAGES
        img_drawn = draw_bboxes_on_image(i,full_path,gt_bboxes,rgb_bboxes_filtered,event_bboxes_filtered,0.5,weights[0],weights[1])

        output_filename = os.path.join('visuals/drawn',sequence_name,'det_'+ filename)
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        cv2.imwrite(output_filename, img_drawn)

    print("Finished processing all images.")

def process_drawn_images_v2(sequence_name,gt_bboxes,rgb_bboxes_filtered,event_bboxes_filtered,hybrid_bboxes,tracked_by_frame,weights=[0.5,0.5],type_data='none'):
        # Define the folder containing your images
    image_folder = os.path.join('data/rgb',sequence_name,'images/left/distorted') 

    # List all files in the directory
    all_files = os.listdir(image_folder)

    # Filter for image files (you might need to expand this list)
    image_extensions = ('.png')
    image_files = sorted([f for f in all_files if f.lower().endswith(image_extensions)])

    if type_data == 'rgb':
        label = 'RGB'
    elif type_data == 'event':
        label = 'Event'
    elif type_data == 'hybrid':
        label = 'Hybrid'
    elif type_data == 'tracked': 
        label = 'Tracked'
    else:
        label = 'None'



    # Loop through the sorted image files
    for i, filename in tqdm(enumerate(image_files[:-1]), total=len(image_files) - 1):
        full_path = os.path.join(image_folder, filename)

        # THIS FUNCTIONS HAS THE PARAMETERS FOR THE WEIGHTS OF THE ALPHAS OF THE IMAGES
        img_drawn = draw_bboxes_on_image_v2(i,full_path,gt_bboxes,rgb_bboxes_filtered,event_bboxes_filtered,hybrid_bboxes,tracked_by_frame,0.5,weights[0],weights[1])
        cv2.putText(img_drawn, label, (20,35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 5)  # black outline
        cv2.putText(img_drawn, label, (20,35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)  # white fill
        output_filename = os.path.join('visuals/drawn',sequence_name,type_data,filename)
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        cv2.imwrite(output_filename, img_drawn)

    print("Finished processing all images.")

#--------- FUSION ------------

def convert_to_wbf_input(detections, image_width, image_height):
    boxes, scores, labels = [], [], []
    for det in detections:
        x1, y1, x2, y2, score, label = det
        boxes.append([x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height])
        scores.append(score)
        labels.append(int(label))
    return boxes, scores, labels

def mark_used_boxes(fused_boxes, orig_boxes, iou_thr):
    used = [False] * len(orig_boxes)
    for fbox in fused_boxes:
        for i, obox in enumerate(orig_boxes):
            # Calculate IoU
            ixmin = max(fbox[0], obox[0])
            iymin = max(fbox[1], obox[1])
            ixmax = min(fbox[2], obox[2])
            iymax = min(fbox[3], obox[3])
            iw = max(ixmax - ixmin, 0)
            ih = max(iymax - iymin, 0)
            inter = iw * ih
            area_f = (fbox[2] - fbox[0]) * (fbox[3] - fbox[1])
            area_o = (obox[2] - obox[0]) * (obox[3] - obox[1])
            union = area_f + area_o - inter
            iou = inter / union if union > 0 else 0
            if iou >= iou_thr:
                used[i] = True
                break
    return used

def new_weighted_fusion(rgb_filtered_bboxes, event_filtered_bboxes, weights=[0.5, 0.5]):
    merged_by_frame = defaultdict(list)
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480
    IOU_THR = 0.5
    SKIP_THR = 0.0

    for frame_idx in tqdm(set(rgb_filtered_bboxes.keys()).union(event_filtered_bboxes.keys())):
        rgb_dets = rgb_filtered_bboxes.get(frame_idx, [])
        event_dets = event_filtered_bboxes.get(frame_idx, [])

        print(f"Processing frame {frame_idx} with {len(rgb_dets)} RGB and {len(event_dets)} Event detections")

        rgb_boxes, rgb_scores, rgb_labels = convert_to_wbf_input(rgb_dets, IMAGE_WIDTH, IMAGE_HEIGHT)
        event_boxes, event_scores, event_labels = convert_to_wbf_input(event_dets, IMAGE_WIDTH, IMAGE_HEIGHT)

        boxes_list = [rgb_boxes, event_boxes]
        scores_list = [rgb_scores, event_scores]
        labels_list = [rgb_labels, event_labels]

        # 🔧 Step 1: Fuse
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            iou_thr=IOU_THR, skip_box_thr=SKIP_THR, weights=weights
        )

        # Convert fused boxes back to absolute coords
        detections_fused = []
        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            x1 = box[0] * IMAGE_WIDTH
            y1 = box[1] * IMAGE_HEIGHT
            x2 = box[2] * IMAGE_WIDTH
            y2 = box[3] * IMAGE_HEIGHT
            detections_fused.append([x1, y1, x2, y2, score, int(label)])

        # 🔧 Step 2: Add all unmatched boxes from RGB and Event
        all_orig_dets = rgb_dets + event_dets
        all_fused_boxes = np.array([[b[0]*IMAGE_WIDTH, b[1]*IMAGE_HEIGHT, b[2]*IMAGE_WIDTH, b[3]*IMAGE_HEIGHT] for b in fused_boxes])
        
        for det in all_orig_dets:
            box = np.array(det[:4])
            if len(all_fused_boxes) == 0:
                ious = np.array([])
            else:
                ious = compute_ious(box, all_fused_boxes)

            if len(ious) == 0 or np.max(ious) < IOU_THR:
                detections_fused.append(det)

        # Final NMS
        merged_by_frame[frame_idx] = detections_fused

        print(f"Frame {frame_idx} processed: {len(merged_by_frame[frame_idx])} detections after fusion")

    return merged_by_frame


def weighted_fusion(rgb_filtered_bboxes,event_filtered_bboxes,weights=[0.5,0.5]):

        merged_by_frame = defaultdict(list)
        IMAGE_WIDTH = 640
        IMAGE_HEIGHT = 480
        IOU_THR = 0.5
        SKIP_THR = 0.0
        
        for frame_idx in tqdm(set(rgb_filtered_bboxes.keys()).union(event_filtered_bboxes.keys())):
            rgb_dets = rgb_filtered_bboxes.get(frame_idx, [])
            event_dets = event_filtered_bboxes.get(frame_idx, [])

            rgb_boxes, rgb_scores, rgb_labels = convert_to_wbf_input(rgb_dets, IMAGE_WIDTH, IMAGE_HEIGHT)
            event_boxes, event_scores, event_labels = convert_to_wbf_input(event_dets, IMAGE_WIDTH, IMAGE_HEIGHT)

            boxes_list = [rgb_boxes, event_boxes]
            scores_list = [rgb_scores, event_scores]
            labels_list = [rgb_labels, event_labels]

            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list,
                iou_thr=IOU_THR, skip_box_thr=SKIP_THR, weights=weights, conf_type='avg'
            )

            # Convert fused boxes back to absolute coords
            detections_fused = []
            for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
                x1 = box[0] * IMAGE_WIDTH
                y1 = box[1] * IMAGE_HEIGHT
                x2 = box[2] * IMAGE_WIDTH
                y2 = box[3] * IMAGE_HEIGHT
                detections_fused.append([x1, y1, x2, y2, score, int(label)])

            if len(fused_boxes) == 0:
                detections_fused = rgb_dets + event_dets
            else:
                # Add back unused original boxes
                all_orig_boxes = rgb_boxes + event_boxes
                all_orig_dets = rgb_dets + event_dets
                used_flags = mark_used_boxes(fused_boxes, all_orig_boxes, iou_thr=IOU_THR)

                for used, det in zip(used_flags, all_orig_dets):
                    if not used and det[4] >= SKIP_THR:
                        detections_fused.append(det)

            """ # Mark used boxes from original detections
            all_orig_boxes = rgb_boxes + event_boxes
            all_orig_dets = rgb_dets + event_dets
            used_flags = mark_used_boxes(fused_boxes, all_orig_boxes, iou_thr=IOU_THR)

            # Add unmatched original boxes
            for used, det in zip(used_flags, all_orig_dets):
                if not used and det[4] >= SKIP_THR:
                    detections_fused.append(det) """

            merged_by_frame[frame_idx] = nms(detections_fused, iou_threshold=0.5)

        return merged_by_frame

def compute_ious(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union_area = box_area + boxes_area - inter_area
    return inter_area / np.maximum(union_area, 1e-6)

def nms(detections, iou_threshold=0.5):
    if not detections:
        return []

    detections = sorted(detections, key=lambda x: x[4], reverse=True)  # Sort by score
    keep = []
    boxes = np.array([det[:4] for det in detections])

    while detections:
        curr = detections.pop(0)
        keep.append(curr)

        if not detections:
            break

        rest_boxes = np.array([d[:4] for d in detections])
        ious = compute_ious(np.array(curr[:4]), rest_boxes)
        detections = [d for i, d in enumerate(detections) if ious[i] < iou_threshold]

    return keep

def nms_fusion(rgb_filtered_bboxes, event_filtered_bboxes):
    merged_by_frame = defaultdict(list)
    IOU_THR = 0.5

    for frame_idx in tqdm(set(rgb_filtered_bboxes.keys()).union(event_filtered_bboxes.keys())):
        rgb_dets = rgb_filtered_bboxes.get(frame_idx, [])
        event_dets = event_filtered_bboxes.get(frame_idx, [])

        all_dets = rgb_dets + event_dets  # Both in absolute coordinates
        merged_by_frame[frame_idx] = nms(all_dets, iou_threshold=IOU_THR)

    return merged_by_frame

# ------------------ TRACKING ----------------------------
def tracking_bytetrack(merged_bboxes, sequence_name,lost_track_buffer=20,frame_rate=20,minimum_matching_threshold=0.8,minimum_consecutive_frames=3):
    """
    Perform tracking using ByteTrack on precomputed detections.

    Args:
        merged_bboxes (dict): {frame_idx: [[x1, y1, x2, y2, score, class_id], ...]}
        sequence_name (str): Name of the sequence to locate frames.

    Returns:
        dict: tracked_bboxes[frame_idx] = [[x1, y1, x2, y2, track_id, class_id], ...]
    """
    byte_tracker = sv.ByteTrack(lost_track_buffer=lost_track_buffer,frame_rate=frame_rate,minimum_matching_threshold=minimum_matching_threshold,minimum_consecutive_frames=minimum_consecutive_frames)
    tracked_bboxes = defaultdict(list)

    for frame_idx in tqdm(sorted(merged_bboxes.keys())):
        frame_path = f"data/rgb/{sequence_name}/images/left/distorted/{frame_idx:06d}.png"
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"⚠️ Frame {frame_idx} not found at {frame_path}")
            continue

        dets = merged_bboxes[frame_idx]
        if len(dets) == 0:
            continue

        boxes = [d[:4] for d in dets]
        scores = [d[4] for d in dets]
        class_ids = [int(d[5]) for d in dets]

        detections = sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            confidence=np.array(scores, dtype=np.float32),
            class_id=np.array(class_ids, dtype=int)
        )

        tracked_detections = byte_tracker.update_with_detections(detections)

        for i in range(len(tracked_detections)):
            x1, y1, x2, y2 = tracked_detections.xyxy[i]
            track_id = tracked_detections.tracker_id[i]
            cls_id = tracked_detections.class_id[i]

            if track_id is None or track_id == -1:
                continue

            tracked_bboxes[frame_idx].append([
                int(x1), int(y1), int(x2), int(y2),
                int(track_id), int(cls_id)
            ])

    return tracked_bboxes

def tracking(merged_bboxes,sequence_name):

    tracker = DeepSort(
    max_age=5,
    n_init=2,
    nms_max_overlap=0.5,
    max_cosine_distance=0.2,
    nn_budget=100,
    override_track_class=None,
    embedder='clip_RN50x4',
    half=False,
    bgr=True,
    embedder_gpu=True,
    polygon=False,
    today=False  
    )

    tracked_bboxes = defaultdict(list)

    for frame_idx in tqdm(sorted(merged_bboxes.keys())):
        frame_tracks = []  # ✅ Important: reset for each frame

        dets = merged_bboxes[frame_idx]
        input_dets = [([d[0], d[1], d[2] - d[0], d[3] - d[1]], d[4], d[5]) for d in dets]

        frame = cv2.imread(f"data/rgb/{sequence_name}/images/left/distorted/{frame_idx:06d}.png")
        if frame is None:
            print(f"Frame {frame_idx} not found")
            continue

        h, w = frame.shape[:2]
        tracks = tracker.update_tracks(input_dets, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            ltrb = track.to_ltrb(orig=True)
            if ltrb is None:
                continue  # skip if no matched detection

            x1, y1, x2, y2 = map(int, ltrb)

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            cls = track.get_det_class()
            track_id = track.track_id
            confidence = 1

            frame_tracks.append([x1, y1, x2, y2, track_id, cls])

            #print(f"Frame {frame_idx} — ID {track_id} — BBox ({x1}, {y1}, {x2}, {y2}) — Class {cls}")

        tracked_bboxes[frame_idx] = frame_tracks  # ✅ Now correct

    return tracked_bboxes,tracker

# Configure ByteTrack + ReID behavior


def draw_tracking_bboxes_on_image(idx, img_path, tracked_by_frame,frame_weight=0.75,event_weight=0.25):
    frame = cv2.imread(img_path)

    # this is to get the sequence_name
    image_path = Path(img_path)
    sequence_name = image_path.parts[2]

    frame = get_event_frame_rgb_dominant(idx, frame, sequence_name, downsample=True, split='test',
                                         frame_weight=frame_weight, event_weight=event_weight)
    if frame is None:
        raise ValueError(f"Frame could not be loaded: {img_path}")

    detections = tracked_by_frame.get(idx, [])

    for det in detections:
        x1, y1, x2, y2, track_id, cls = det

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id} | Cls {cls}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return frame

def process_drawn_tracked_images(sequence_name, tracked_by_frame,frame_weight=0.75,event_weight=0.25):
    # Define the folder containing your images
    image_folder = os.path.join('data/rgb', sequence_name, 'images/left/distorted')

    # List all files in the directory
    all_files = os.listdir(image_folder)

    # Filter for image files
    image_extensions = ('.png',)
    image_files = sorted([f for f in all_files if f.lower().endswith(image_extensions)])

    # Loop through the sorted image files
    for i, filename in tqdm(enumerate(image_files), total=len(image_files)):
        full_path = os.path.join(image_folder, filename)

        try:
            img_drawn = draw_tracking_bboxes_on_image(i, full_path, tracked_by_frame,frame_weight,event_weight)
        except ValueError as e:
            print(f"Skipping frame {i}: {e}")
            continue

        # Define output path
        output_filename = os.path.join('visuals/tracked', sequence_name, 'trk_' + filename)
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        cv2.imwrite(output_filename, img_drawn)

    print("Finished processing all tracked images.")

# ------------------ METRICS -----------------------

def evaluate_motmetrics(gt_by_frame, tracked_by_frame, iou_threshold=0.5, debug=False):
    """
    Evaluates MOT metrics per class and prints a combined summary.
    Assumes:
        gt_by_frame[frame_id] = [[x1, y1, x2, y2, class_id, track_id], ...]
        tracked_by_frame[frame_id] = [[x1, y1, x2, y2, track_id, class_id], ...]
    """
    mh = mm.metrics.create()
    class_summaries = {}

    all_classes = set()
    for frame_dets in gt_by_frame.values():
        all_classes.update(det[5] for det in frame_dets)

    

    for cls_id in sorted(all_classes):
        acc = mm.MOTAccumulator(auto_id=True)
        all_frames = sorted(set(gt_by_frame.keys()) | set(tracked_by_frame.keys()))

        for frame_id in all_frames:
            gt_dets = [det for det in gt_by_frame.get(frame_id, []) if det[5] == cls_id]
            trk_dets = [det for det in tracked_by_frame.get(frame_id, []) if det[5] == cls_id]

            gt_ids = [det[6] for det in gt_dets]
            trk_ids = [det[4] for det in trk_dets]

            gt_boxes = np.array([det[:4] for det in gt_dets])
            trk_boxes = np.array([det[:4] for det in trk_dets])

            def to_xywh(boxes):
                return np.array([[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes])

            if debug:
                print(f"\n=== Frame {frame_id} | Class {cls_id} ===")
                print(f"GT IDs: {gt_ids}")
                print(f"TRK IDs: {trk_ids}")
                print(f"GT Boxes: {gt_boxes}")
                print(f"TRK Boxes: {trk_boxes}")

            if len(gt_boxes) == 0 or len(trk_boxes) == 0:
                distances = np.ones((len(gt_boxes), len(trk_boxes))) * np.nan
                if debug:
                    print("⚠️ Empty boxes: skipping IoU calculation.")
            else:
                distances = mm.distances.iou_matrix(
                    to_xywh(gt_boxes),
                    to_xywh(trk_boxes),
                    max_iou=iou_threshold
                )
                if debug:
                    print(f"IoU Distances:\n{distances}")

            acc.update(gt_ids, trk_ids, distances)

        summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name=f'class_{cls_id}')
        class_summaries[cls_id] = summary

    # Print summaries
    #merged_summary = mm.io.merge_summary(class_summaries, generate_overall=True)
    #print(mm.io.render_summary(merged_summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

    # Print per-class summaries
    for cls_id, summary in class_summaries.items():
        print(f"\n📊 Summary for Class {cls_id}:")
        print(mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        ))
        
    return class_summaries

def create_collage_video(folder_paths, output_video_path, output_resolution=(1920, 1080), fps=30):
    """
    Creates a video from images in four folders, arranging them into a 2x2 collage.

    Args:
        folder_paths (list): A list of four strings, each being the path to an image folder.
                             Images within each folder should be numerically ordered (e.g., 001.png, 002.png).
        output_video_path (str): The full path and filename for the output video (e.g., 'my_collage_video.mp4').
        output_resolution (tuple): A tuple (width, height) for the desired video resolution.
        fps (int): Frames per second for the output video.
    """

    if len(folder_paths) != 4:
        print("Error: Please provide exactly four folder paths.")
        return

    # Get sorted list of image files from each folder
    image_files_per_folder = []
    for folder_path in folder_paths:
        if not os.path.isdir(folder_path):
            print(f"Error: Folder not found: {folder_path}")
            return
        
        # Filter for common image extensions
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        # --- CORRECTED SORTING LOGIC ---
        files.sort(key=lambda f: (
            int(os.path.splitext(f)[0]) if os.path.splitext(f)[0].isdigit() else float('inf'), # Sort by int if possible
            f # Fallback to string sort if not a pure number
        ))
        # --- END CORRECTED SORTING LOGIC ---
        
        image_files_per_folder.append([os.path.join(folder_path, f) for f in files])

    # Determine the minimum number of images across all folders
    min_images = min(len(files) for files in image_files_per_folder)
    if min_images == 0:
        print("Error: No images found in one or more specified folders.")
        return

    print(f"Found {min_images} sets of images to process.")

    # Calculate individual image dimensions for the collage
    img_width = output_resolution[0] // 2
    img_height = output_resolution[1] // 2

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
    out = cv2.VideoWriter(output_video_path, fourcc, fps, output_resolution)

    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_video_path} Check path or codec.")
        print("Common codecs: 'mp4v' for .mp4, 'XVID' for .avi")
        print("You might need to install additional codecs or ensure OpenCV is built with FFMPEG support.")
        return

    for i in tqdm(range(min_images), desc="Creating video frames"):
        collage_img = Image.new('RGB', output_resolution)
        
        # Load and resize images for the current frame
        images_for_frame = []
        for folder_idx in range(4):
            try:
                img = Image.open(image_files_per_folder[folder_idx][i]).convert('RGB')
                img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
                images_for_frame.append(img)
            except Exception as e:
                print(f"Warning: Error loading/processing image {image_files_per_folder[folder_idx][i]}: {e}")
                # Fallback to a black image if an error occurs to keep video consistent
                images_for_frame.append(Image.new('RGB', (img_width, img_height), color = 'black'))


        # Paste images into the collage
        collage_img.paste(images_for_frame[0], (0, 0))             # Top-left
        collage_img.paste(images_for_frame[1], (img_width, 0))     # Top-right
        collage_img.paste(images_for_frame[2], (0, img_height))    # Bottom-left
        collage_img.paste(images_for_frame[3], (img_width, img_height)) # Bottom-right

        # Convert PIL image to OpenCV format (NumPy array)
        opencv_frame = cv2.cvtColor(np.array(collage_img), cv2.COLOR_RGB2BGR)
        out.write(opencv_frame)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{min_images} frames...")

    out.release()
    print(f"Video created successfully at: {output_video_path}")

# --- Your working function (slightly modified for clarity) ---
def create_video_from_frames_v2(input_folder, output_video_path, fps=20):
    """
    Creates a video from image frames located in a single input folder.
    This function is known to work with 'mp4v' on your system.
    """
    images = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not images:
        print(f"No image files found in {input_folder}.")
        return False # Indicate failure

    first_frame_path = os.path.join(input_folder, images[0])
    first_frame = cv2.imread(first_frame_path)
    
    if first_frame is None:
        print(f"Error: Could not read the first frame from {first_frame_path}. Check file integrity/permissions.")
        return False

    height, width, _ = first_frame.shape
    size = (width, height)

    # Use 'mp4v' which you confirmed works for this function
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_video_path} using 'mp4v' codec.")
        print("This specific error might indicate issues with the output path/permissions or unexpected codec problems.")
        return False # Indicate failure

    for img_name in tqdm(images, desc=f"Encoding video from {input_folder}"):
        img_path = os.path.join(input_folder, img_name)
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"\nWarning: Could not read frame {img_path}. Skipping.")
            continue # Skip corrupted/unreadable frames

        # Resize if necessary (though collage should already be correct size)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, size)
            
        out.write(frame)

    out.release()
    print(f"Video saved to {output_video_path}")
    return True # Indicate success

# --- Modified create_collage_video function ---
def create_collage_video_v2(folder_paths, output_video_path, output_resolution=(1920, 1080), fps=20):
    """
    Creates a video from images in four folders, arranging them into a 2x2 collage.
    It generates temporary collage frames and then uses `create_video_from_frames`
    to produce the final MP4 video.
    """

    if len(folder_paths) != 4:
        print("Error: Please provide exactly four folder paths.")
        return

    # Create a temporary directory to store the collage frames
    temp_frames_dir = "temp_collage_frames_" + str(os.getpid()) # Unique name
    os.makedirs(temp_frames_dir, exist_ok=True)
    print(f"Temporary collage frames will be saved in: {temp_frames_dir}")

    # Get sorted list of image files from each folder
    image_files_per_folder = []
    for folder_path in folder_paths:
        if not os.path.isdir(folder_path):
            print(f"Error: Folder not found: {folder_path}")
            shutil.rmtree(temp_frames_dir) # Clean up temp
            return
        
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        files.sort(key=lambda f: (
            int(os.path.splitext(f)[0]) if os.path.splitext(f)[0].isdigit() else float('inf'),
            f
        ))
        
        image_files_per_folder.append([os.path.join(folder_path, f) for f in files])

    # Determine the minimum number of images across all folders
    min_images = min(len(files) for files in image_files_per_folder)
    if min_images == 0:
        print("Error: No images found in one or more specified folders.")
        shutil.rmtree(temp_frames_dir) # Clean up temp
        return

    print(f"Found {min_images} sets of images to process to create collage frames.")

    # Calculate individual image dimensions for the collage
    img_width = output_resolution[0] // 2
    img_height = output_resolution[1] // 2

    # Generate and save collage frames
    for i in tqdm(range(min_images), desc="Generating collage frames"):
        collage_img = Image.new('RGB', output_resolution)
        
        images_for_frame = []
        for folder_idx in range(4):
            try:
                img = Image.open(image_files_per_folder[folder_idx][i]).convert('RGB')
                img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
                images_for_frame.append(img)
            except Exception as e:
                print(f"\nWarning: Error loading/processing image {image_files_per_folder[folder_idx][i]}: {e}", end='\r')
                images_for_frame.append(Image.new('RGB', (img_width, img_height), color = 'black'))


        collage_img.paste(images_for_frame[0], (0, 0))             # Top-left
        collage_img.paste(images_for_frame[1], (img_width, 0))     # Top-right
        collage_img.paste(images_for_frame[2], (0, img_height))    # Bottom-left
        collage_img.paste(images_for_frame[3], (img_width, img_height)) # Bottom-right

        # Save the collage frame to the temporary directory
        temp_frame_path = os.path.join(temp_frames_dir, f"frame_{i:06d}.png") # 6-digit padding
        collage_img.save(temp_frame_path)

    print(f"Finished generating {min_images} collage frames.")

    # --- Use your working create_video_from_frames function ---
    print(f"Now creating final video using generated frames from {temp_frames_dir}...")
    success = create_video_from_frames_v2(temp_frames_dir, output_video_path, fps)

    # Clean up the temporary folder
    if os.path.exists(temp_frames_dir):
        print(f"Cleaning up temporary directory: {temp_frames_dir}")
        shutil.rmtree(temp_frames_dir)
    
    if success:
        print(f"Overall video creation complete: {output_video_path}")
    else:
        print("Video creation failed in the final encoding step.")

def adjust_brightness_add(image, value=10):
    """
    Adjusts image brightness by adding a constant value to all pixels.
    Values are clamped between 0 and 255.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Add the value to the V (Value/Brightness) channel
    v_new = cv2.add(v, value)
    v_new = np.clip(v_new, 0, 255).astype(np.uint8) # Clamp values

    final_hsv = cv2.merge([h, s, v_new])
    brightened_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return brightened_image
        
    


  







            



            



            
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
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as T


class ResNet50Embedder:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]).to(self.device)
        self.backbone.eval()

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def preprocess(self, img_bgr):
        img = cv2.resize(img_bgr, (64, 128))  # Resize as in re-ID datasets
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
        img_tensor = self.normalize(img_tensor)
        return img_tensor.unsqueeze(0).to(self.device)

    def __call__(self, crop_bgr):
        input_tensor = self.preprocess(crop_bgr)
        with torch.no_grad():
            embedding = self.backbone(input_tensor).squeeze()  # 2048-dim
        return embedding.cpu().numpy()

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

def filter_pred_by_confidence(preds_by_frame,conf_threshold):

    if conf_threshold > 1 or conf_threshold < 0:
        assert "Confidence Threshold must be between 0 and 1"
        
    preds_filtered = {
        frame: [det for det in dets if det[4] >= conf_threshold]
        for frame, dets in preds_by_frame.items()
    }

    return preds_filtered

def generate_metrics_csv(sequence_name, rgb_preds_by_frame, event_preds_by_frame, confidence_thresholds, gt_bboxes):

    def write_metrics_file(filename, preds_by_frame):
        with open(filename, mode='w', newline='') as file:
            csv_writer = csv.writer(file, delimiter=';')
            csv_writer.writerow(['conf', 'class', 'mAP', 'mAP50', 'P50', 'R50','mAP75', 'P75', 'R75'])

            for conf in confidence_thresholds:
                preds_filtered = filter_pred_by_confidence(preds_by_frame, conf)
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

def draw_bboxes_on_image(idx,img_path, gt_bboxes, rgb_bboxes, event_bboxes, confidence=0.5,frame_weight=0.75,event_weight=0.75):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Image could not be loaded: {img_path}")
    
    # this is to get the sequence_name
    image_path = Path(img_path)
    sequence_name = image_path.parts[2]
    
    
    image = get_event_frame_rgb(idx,image,sequence_name=sequence_name,frame_weight=frame_weight,event_weight=event_weight)

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
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 182, 255), 2)
                cv2.putText(image, f"P:{int(cls)} {conf:.2f}", (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 182, 255), 2)

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

def process_drawn_images(sequence_name,gt_bboxes,rgb_bboxes_filtered,event_bboxes_filtered):
        # Define the folder containing your images
    image_folder = os.path.join('data/rgb',sequence_name,'images/left/distorted') # <--- IMPORTANT: Change this to your actual folder path

    # List all files in the directory
    all_files = os.listdir(image_folder)

    # Filter for image files (you might need to expand this list)
    image_extensions = ('.png')
    image_files = sorted([f for f in all_files if f.lower().endswith(image_extensions)])


    # Loop through the sorted image files
    for i, filename in tqdm(enumerate(image_files[:-1]), total=len(image_files) - 1):
        full_path = os.path.join(image_folder, filename)

        # THIS FUNCTIONS HAS THE PARAMETERS FOR THE WEIGHTS OF THE ALPHAS OF THE IMAGES
        img_drawn = draw_bboxes_on_image(i,full_path,gt_bboxes,rgb_bboxes_filtered,event_bboxes_filtered,0.5)

        output_filename = os.path.join('visuals/drawn',sequence_name,'det_'+ filename)
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

            # Mark used boxes from original detections
            all_orig_boxes = rgb_boxes + event_boxes
            all_orig_dets = rgb_dets + event_dets
            used_flags = mark_used_boxes(fused_boxes, all_orig_boxes, iou_thr=IOU_THR)

            # Add unmatched original boxes
            for used, det in zip(used_flags, all_orig_dets):
                if not used and det[4] >= SKIP_THR:
                    detections_fused.append(det)

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

def tracking(merged_bboxes,sequence_name):

    tracker = DeepSort(
    max_age=2,
    n_init=4,
    nms_max_overlap=0.5,
    max_cosine_distance=0.3,
    nn_budget=50,
    override_track_class=None,
    embedder='clip_RN101',
    half=True,
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

def draw_tracking_bboxes_on_image(idx, img_path, tracked_by_frame, tracker):
    frame = cv2.imread(img_path)
    if frame is None:
        raise ValueError(f"Frame could not be loaded: {img_path}")

    detections = tracked_by_frame.get(idx, [])

    input_dets = [([d[0], d[1], d[2] - d[0], d[3] - d[1]], d[4], d[5]) for d in detections]

    tracks = tracker.update_tracks(input_dets, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        ltrb = track.to_ltrb(orig=True)
        if ltrb is None:
            continue

        x1, y1, x2, y2 = map(int, ltrb)
        track_id = track.track_id
        cls = track.get_det_class()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id} | Cls {cls}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return frame

def process_drawn_tracked_images(sequence_name, tracked_by_frame, tracker):
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
            img_drawn = draw_tracking_bboxes_on_image(i, full_path, tracked_by_frame, tracker)
        except ValueError as e:
            print(f"Skipping frame {i}: {e}")
            continue

        # Define output path
        output_filename = os.path.join('visuals/tracked', sequence_name, 'trk_' + filename)
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        cv2.imwrite(output_filename, img_drawn)

    print("Finished processing all tracked images.")
                











            



            



            
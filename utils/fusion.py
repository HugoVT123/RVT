from collections import defaultdict
import torch
from tqdm import tqdm
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
import torchvision.ops as ops  # for NMS


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
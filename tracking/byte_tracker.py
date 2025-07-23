import supervision as sv
from tqdm import tqdm
import cv2
from collections import defaultdict
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

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

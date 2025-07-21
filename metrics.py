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
confidence_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45 , 0.5, 0.55, 0.60, 0.65 , 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
iou_thresholds = [0.5, 0.75]

get_tracking_metrics = True

# FOR TESTING PURPOSES ONLY
#all_sequences = ["thun_01_b"] # DELETE BEFORE DOING AUTOMATIZATION

sequences_to_remove = ["interlaken_01_a","zurich_city_00_b"]

all_sequences = [item for item in all_sequences if item not in sequences_to_remove]

""" all_sequences = ["interlaken_00_b",
                 "zurich_city_12_a",
                 "zurich_city_13_a",
                 "zurich_city_14_a",
                 "zurich_city_15_a"] """

csv_path = "metrics/metrics.csv"

write_header = not os.path.exists(csv_path)

with open(csv_path, mode='a', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    if write_header:
        writer.writerow(["Sequence", "Type", "class","IDF1", "IDP", "IDR", "Rcll",
                          "Prcn", "GT", "MT", "PT", "ML", "FP", "FN", "IDs", "FM", "MOTA", "MOTP", "IDt", "IDa", "IDm"])


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

        
        rgb_bboxes_filtered = filter_pred_by_confidence(rgb_bboxes, {0: 0.05, 1: 0.1})
        event_bboxes_filtered = filter_pred_by_confidence(event_bboxes,{0: 0.001, 1: 0.001})

        


        # DELETE THIS AFTER GETTIN THE IMAGES
        #gt_bboxes = {}
        #event_bboxes_filtered = {}
        #rgb_bboxes_filtered = {}


        print('------------------RGB & Event Detections----------------')
        
        #generate_metrics_csv(sequence_name,rgb_bboxes_filtered,event_bboxes_filtered,confidence_thresholds,gt_bboxes)

        #process_drawn_images(sequence_name,gt_bboxes,rgb_bboxes_filtered,event_bboxes_filtered,weights=[1.0,0.35])

        #create_video_from_frames(os.path.join('visuals/drawn',sequence_name),os.path.join('visuals/videos',f'{sequence_name}.mp4'))

        #---------------------- MERGE DETETCTIONS -----------------
        print('------------------MERGED DETETCTIONS----------------')

        merged_bboxes = weighted_fusion(rgb_bboxes_filtered,event_bboxes_filtered,weights=[0.7,0.3]) # <----- WEIGHTS

        #merged_bboxes_filtered = filter_pred_by_confidence(merged_bboxes, {0: 0.5, 1: 0.5})

        #generate_metrics_csv(sequence_name,merged_bboxes,merged_bboxes,confidence_thresholds,gt_bboxes)
        #process_drawn_images(sequence_name,gt_bboxes,merged_bboxes,{},weights=[1.0,0.35])

        #create_video_from_frames(os.path.join('visuals/drawn',sequence_name),os.path.join('visuals/videos',f'{sequence_name}_fused.mp4'))


        #-------------------- TRACKING -----------------------------------

        if get_tracking_metrics:
            print('------------------TRACKING----------------')


            # Perform tracking
            rgb_tracked_by_frame = tracking_bytetrack(rgb_bboxes_filtered, sequence_name)
            event_tracked_by_frame = tracking_bytetrack(event_bboxes_filtered, sequence_name)
            merged_tracked_by_frame = tracking_bytetrack(merged_bboxes, sequence_name)

            # Evaluate tracking
            tracking_results = {
                "rgb": evaluate_motmetrics(gt_by_frame, rgb_tracked_by_frame),
                "event": evaluate_motmetrics(gt_by_frame, event_tracked_by_frame),
                "hybrid": evaluate_motmetrics(gt_by_frame, merged_tracked_by_frame)
            }

            #process_drawn_tracked_images(sequence_name,merged_tracked_by_frame,frame_weight=1.0,event_weight=0.35)

            # Write results
            for tipo, class_summaries in tracking_results.items():
                for cls_id in class_summaries.keys():
                    c_df = class_summaries[cls_id]  # Esto es un DataFrame con shape (1, 18)
                    row_key = c_df.index[0]  # Ej: 'class_0'
                    c = c_df.loc[row_key].to_dict()

                    
                    row = [
                        sequence_name,
                        tipo,
                        cls_id,
                        round(c.get("idf1", 0), 6),
                        round(c.get("idp", 0), 6),
                        round(c.get("idr", 0), 6),
                        round(c.get("recall", 0), 6),
                        round(c.get("precision", 0), 6),
                        int(c.get("num_unique_objects", 0)),
                        int(c.get("mostly_tracked", 0)),
                        int(c.get("partially_tracked", 0)),
                        int(c.get("mostly_lost", 0)),
                        int(c.get("num_false_positives", 0)),
                        int(c.get("num_misses", 0)),
                        int(c.get("num_switches", 0)),
                        int(c.get("num_fragmentations", 0)),
                        round(c.get("mota", 0), 6),
                        round(c.get("motp", 0), 6),
                        int(c.get("num_transfer", 0)),
                        int(c.get("num_ascend", 0)),
                        int(c.get("num_migrate", 0)),
                    ]
                    writer.writerow(row)














    





    

    

    

    



    



    
    
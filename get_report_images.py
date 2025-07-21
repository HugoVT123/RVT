import os
import numpy as np
from tqdm import tqdm
import os
from scripts.my_toolbox import *


SCALE_Y = 360 / 480  # = 0.75

split = "test"
root_dir = os.path.join("data/dsec_proc",split)
all_sequences = [item for item in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, item))]

get_tracking_metrics = False

# FOR TESTING PURPOSES ONLY
all_sequences = ["zurich_city_14_c"] # DELETE BEFORE DOING AUTOMATIZATION

sequences_to_remove = ["interlaken_01_a","zurich_city_00_b"]

all_sequences = [item for item in all_sequences if item not in sequences_to_remove]

""" all_sequences = ["interlaken_00_b",
                 "zurich_city_12_a",
                 "zurich_city_13_a",
                 "zurich_city_14_a",
                 "zurich_city_15_a"] """

video_resolution = (640*2, 480*2) # You can change this
video_fps = 20 # Frames per second


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

    
    rgb_bboxes_filtered = filter_pred_by_confidence(rgb_bboxes, {0: 0.001, 1: 0.001})
    event_bboxes_filtered = filter_pred_by_confidence(event_bboxes,{0: 0.001, 1: 0.001})

    print('------------------RGB Detections----------------')

    empty_bboxes = {}
    
    process_drawn_images_v2(sequence_name=sequence_name,
                            gt_bboxes=gt_bboxes,
                            rgb_bboxes_filtered=rgb_bboxes_filtered,
                            event_bboxes_filtered=empty_bboxes,
                            hybrid_bboxes=empty_bboxes,
                            tracked_by_frame=empty_bboxes,
                            weights=[1.0,0.0],
                            type_data='rgb')

    #create_video_from_frames(os.path.join('visuals/drawn',sequence_name),os.path.join('visuals/videos',f'{sequence_name}.mp4'))

    print('------------------Event Detections----------------')

    process_drawn_images_v2(sequence_name=sequence_name,
                            gt_bboxes=gt_bboxes,
                            rgb_bboxes_filtered=empty_bboxes,
                            event_bboxes_filtered=event_bboxes_filtered,
                            hybrid_bboxes=empty_bboxes,
                            tracked_by_frame=empty_bboxes,
                            weights=[0.0,1.0],
                            type_data='event')

    print('------------------Hybrid Detections----------------')

    merged_bboxes = weighted_fusion(rgb_bboxes_filtered,event_bboxes_filtered,weights=[0.7,0.3]) # <----- WEIGHTS

    process_drawn_images_v2(sequence_name=sequence_name,
                            gt_bboxes=gt_bboxes,
                            rgb_bboxes_filtered=empty_bboxes,
                            event_bboxes_filtered=empty_bboxes,
                            hybrid_bboxes=merged_bboxes,
                            tracked_by_frame=empty_bboxes,
                            weights=[0.5,0.2],
                            type_data='hybrid')

   
    
    print('------------------Tracking ----------------')
       
    tracked_by_frame = tracking_bytetrack(merged_bboxes, sequence_name)
    process_drawn_images_v2(sequence_name=sequence_name,
                            gt_bboxes=gt_bboxes,
                            rgb_bboxes_filtered=empty_bboxes,
                            event_bboxes_filtered=empty_bboxes,
                            hybrid_bboxes=empty_bboxes,
                            tracked_by_frame=tracked_by_frame,
                            weights=[0.5,0.2],
                            type_data='tracked')
    
    folder_1 = os.path.join('visuals/drawn',sequence_name,'rgb')
    folder_2 = os.path.join('visuals/drawn',sequence_name,'event')
    folder_3 = os.path.join('visuals/drawn',sequence_name,'hybrid')
    folder_4 = os.path.join('visuals/drawn',sequence_name,'tracked')

    output_directory = 'visuals/drawn/videos'
    os.makedirs(output_directory, exist_ok=True)

    output_video_name = os.path.join(output_directory,f'{sequence_name}.mp4')

    # --- Run the function ---
    create_collage_video_v2(
        folder_paths=[folder_1, folder_2, folder_3, folder_4],
        output_video_path=output_video_name,
        output_resolution=video_resolution,
        fps=video_fps
    )



    
    
    


   

    

        
            














    





    

    

    

    



    



    
    
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm
from utils.helpers import adjust_brightness_add
from event.blending import get_event_frame_rgb_dominant

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


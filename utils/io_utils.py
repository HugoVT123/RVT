import csv
import cv2
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import shutil

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


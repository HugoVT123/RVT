import os
import numpy as np
from PIL import Image
import cv2
import torch
from rfdetr import RFDETRLarge

# Paths
predictions_folder = "predictions/rgb"
sequences_folder = "data/rgb/test"

# Create predictions folder if it doesn't exist
os.makedirs(predictions_folder, exist_ok=True)

# Detect GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = RFDETRLarge(device=device)

# Get list of sequences
sequences = [name for name in os.listdir(sequences_folder)
             if os.path.isdir(os.path.join(sequences_folder, name))]

# Process each sequence
for sequence in sequences:
    sequence_folder = os.path.join(sequences_folder, sequence)
    input_path = os.path.join(sequence_folder, "images/left/distorted")

    all_predictions = []

    # Sorted list of image files
    image_files = sorted(
        [f for f in os.listdir(input_path) if f.endswith(".png")],
        key=lambda x: int(x.split('.')[0])
    )

    for frame_idx, filename in enumerate(image_files):
        path = os.path.join(input_path, filename)

        # Read image with OpenCV and convert BGR → RGB
        image_cv2 = cv2.imread(path)
        image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

        # Convert to PIL for model input
        image_pil = Image.fromarray(image_rgb)

        # Get predictions
        predictions = model.predict(image_pil, threshold=0.05)

        # Keep only person (1) and car (3)
        desired_class_ids = [1, 3]
        predictions = predictions[np.isin(predictions.class_id, desired_class_ids)]

        # Store detections
        for i in range(len(predictions)):
            all_predictions.append({
                "frame_idx": frame_idx,
                "filename": filename,
                "class_id": int(predictions.class_id[i]),
                "bbox": predictions.xyxy[i].tolist(),  # [x1, y1, x2, y2]
                "confidence": float(predictions.confidence[i])
            })

        print(f"Processed frame {frame_idx + 1}/{len(image_files)}: {filename}")

    # Save detections to .npy file
    predictions_path = os.path.join(predictions_folder, f"{sequence}.npy")
    np.save(predictions_path, all_predictions)
    print(f"Saved detections to {predictions_path}")

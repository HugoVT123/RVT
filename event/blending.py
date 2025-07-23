import numpy as np
import h5py
import os
import cv2

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


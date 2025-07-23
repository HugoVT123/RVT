from typing import Union
import torch as th
import cv2
import numpy as np


def torch_uniform_sample_scalar(min_value: float, max_value: float):
    assert max_value >= min_value, f'{max_value=} is smaller than {min_value=}'
    if max_value == min_value:
        return min_value
    return min_value + (max_value - min_value) * th.rand(1).item()

def clamp(value: Union[int, float], smallest: Union[int, float], largest: Union[int, float]):
    return max(smallest, min(value, largest))

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

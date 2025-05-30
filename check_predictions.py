from data.utils.types import ObjDetOutput
import torch
from modules.utils.detection import EventReprSelector

data = torch.load('predictions/predictions.pt', map_location='cpu')


# Let's say you want to look at the first item
print(type(data))  # Should be a list or similar structure
print(len(data))  # Should be the number of predictions
first_item = data[35]
print(type(first_item))  # Should be dict

# Now access keys using Enum
predictions = first_item[ObjDetOutput.PRED_PROPH]
print(predictions.shape)  # Should be torch.Tensor
print(predictions.dtype)  # Should be torch.float32 or similar


for bbox in predictions:
    print(f"Frame {bbox['t']}: Class {bbox['class_id']} at "
          f"({bbox['x']:.1f}, {bbox['y']:.1f}, {bbox['w']:.1f}, {bbox['h']:.1f}) "
          f"Conf: {bbox['class_confidence']:.2f}")




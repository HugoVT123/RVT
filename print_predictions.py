import numpy as np
import torch
from data.utils.types import ObjDetOutput


data = torch.load("predictions/predictions_1.pt")  # Load the data from the file



# Assume `data` is the list you printed
for item in data:
    labels = item.get(ObjDetOutput.LABELS_PROPH)
    preds = item.get(ObjDetOutput.PRED_PROPH)

    print("LABELS_PROPH:")
    print(labels)
    print("\nPRED_PROPH:")
    print(preds)
    print("\n" + "-"*80 + "\n")
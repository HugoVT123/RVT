import cv2
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig
from data.genx_utils.sequence_for_streaming import SequenceForIter
from utils.evaluation.prophesee.visualize.vis_utils import draw_bboxes

@hydra.main(config_path="config", config_name="mycode", version_base="1.2")
def main(config: DictConfig):
    # Determine the dataset split to use
    split = config.dataset.split  # e.g., "train", "val", or "test"
    dataset_path = Path(config.dataset.path) / split

    # Load the sequence
    sequence = SequenceForIter(
        path=dataset_path,
        ev_representation_name=config.dataset.ev_representation_name,
        sequence_length=config.dataset.sequence_length,
        dataset_type=config.dataset.type,
        downsample_by_factor_2=config.dataset.downsample_by_factor_2
    )

    # Iterate over the sequence
    for idx in range(len(sequence)):
        sample = sequence[idx]
        event_repr = sample["EV_REPR"]
        labels = sample["OBJLABELS_SEQ"]

        # Convert event representation to an image
        event_image = event_repr[0].numpy()  # Assuming the first frame in the sequence
        event_image = (event_image * 255).astype("uint8")  # Normalize to 0-255

        # Draw bounding boxes
        draw_bboxes(event_image, labels, labelmap=("car", "pedestrian"))

        # Save the image
        output_dir = Path("output") / split
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / f"frame_{idx}.png"), event_image)

if __name__ == "__main__":
    main()
# debug_test.py
import time
import numpy as np


# Paths to the generated files
event_repr_timestamps_path = 'data/dsec_proc/test/sequence_name/event_representations_v2/stacked_histogram_dt=50_nbins=10/timestamps_us.npy'  # Replace with your actual path
labels_npz_path = 'data/dsec_proc/test/sequence_name/labels_v2/labels.npz'  # Replace with your actual path

# Load the timestamps
event_repr_timestamps = np.load(event_repr_timestamps_path)
print("Event Representation Timestamps:", event_repr_timestamps)
print("Event Representation Timestamps Shape:", event_repr_timestamps.shape)
print("Event Representation Timestamps Diff:", np.diff(event_repr_timestamps))

# Load the labels
labels_data = np.load(labels_npz_path, allow_pickle=True)
labels = labels_data['labels']
print("\nLabels:", labels)
print("Labels Shape:", labels.shape)

# Print the first few label timestamps
print("\nFirst few label timestamps:", labels['t'][:10])

# Calculate the differences between consecutive timestamps
timestamp_diffs = np.diff(labels['t'])
print("\nTimestamp differences between consecutive labels:", timestamp_diffs)

# Find unique timestamp differences and their counts
unique_diffs, diff_counts = np.unique(timestamp_diffs, return_counts=True)
print("\nUnique timestamp differences and their counts:")
for diff, count in zip(unique_diffs, diff_counts):
    print(f"Difference: {diff}, Count: {count}")

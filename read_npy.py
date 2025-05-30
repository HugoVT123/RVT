import numpy as np

# Replace 'your_file.npy' with the actual path to your .npy file
try:
    data = np.load('data/dummy_dsec/test/thun_00_a_bbox.npy')
    print("\nShape of the array:", data.shape)
    print("Data type of the array:", data.dtype)
except FileNotFoundError:
    print(f"Error: File 'your_file.npy' not found.")
except Exception as e:
    print(f"An error occurred: {e}")
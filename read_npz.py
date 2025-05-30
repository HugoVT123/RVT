import numpy as np

# Replace 'your_file.npz' with the actual path to your file
file_path = 'data/gen4_proc/train/moorea_2019-02-15_001_td_183500000_243500000/labels_v2/labels.npz'
my_file_path = 'data/dummy_dsec_proc/test/thun_00_a/labels_v2/labels.npz'
other_file_path = 'data/dummy_dsec_proc/test/thun_00_a/labels_v2/labels.npz'
predictions = "predictions/predictions.npz"

try:
    # Load the .npz file
    data = np.load(other_file_path,allow_pickle=True)  # allow_pickle=True if the arrays contain objects

    print("DATA TYPE:", type(data))  # Print the type of the loaded data
    # Get the list of array names
    array_names = data.files

    # Print the names
    print(f"Arrays found in '{file_path}':")
    for name in array_names:
        print(name)

    # If you want to see the actual content of the first array (as an example)
    if array_names and False: # Check if there are any arrays
        first_array_name = array_names[0]
        print(f"\nContent of the first array ('{first_array_name}'):")
        print(data[first_array_name])  # Print the data type of the first array
        print("Shape type of the first array:", data[first_array_name].shape)

        second_array_name = array_names[1]
        print(f"\nContent of the second array ('{second_array_name}'):")
        print(data[second_array_name])  # Print the data type of the second array
        print("Shape type of the second array:", data[second_array_name].shape)

    # It's good practice to close the file
    data.close()

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
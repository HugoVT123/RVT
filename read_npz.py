import numpy as np

# Replace 'your_file.npz' with the actual path to your file

predictions = "predictions/preprocess_dsec/interlaken_00_a.npz"
labels = "data/dsec_proc/test/interlaken_00_a/labels_v2/labels.npz"

try:
    # Load the .npz file
    data = np.load(labels,allow_pickle=True)  # allow_pickle=True if the arrays contain objects

    print("DATA TYPE:", type(data))  # Print the type of the loaded data
    # Get the list of array names
    array_names = data.files

    # Print the names
    for name in array_names:
        print(name)

    # If you want to see the actual content of the first array (as an example)
    if array_names: # Check if there are any arrays
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
    print(f"File not found: {labels}")
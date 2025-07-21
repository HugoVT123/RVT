import numpy as np

def txt_to_npy(txt_path, npy_path):
    # Define the desired structured dtype
    dtype = [
        ('t', '<u8'),                    # timestamp
        ('x', '<f4'),                   # x
        ('y', '<f4'),                   # y
        ('w', '<f4'),                   # width
        ('h', '<f4'),                   # height
        ('class_id', 'u1'),             # class ID
        ('class_confidence', '<f4'),    # confidence
        ('track_id', '<u4')             # track ID
    ]

    data_list = []

    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 7:
                continue  # Skip malformed lines

            # Parse and format the data
            timestamp = int(parts[0])
            class_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            track_id = int(parts[6])
            class_conf = 1  # You can set a default or compute this differently

            data_list.append((timestamp, x, y, w, h, class_id, class_conf, track_id))

    # Convert to structured NumPy array
    structured_array = np.array(data_list, dtype=dtype)

    # Save to .npy
    np.save(npy_path, structured_array)
    return structured_array
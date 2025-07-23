import hdf5plugin  # Necesario si el archivo usa compresión especial
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import io
from utils.helpers import get_number_of_rgb_frames
from tqdm import tqdm

def adaptive_process_and_save_event_tensor_sequence(
    h5_file, h5_output_path, t_min, new_timestamps, T=10, H=360, W=640, verbose=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print(f"Using device: {device}")
    x = h5_file['events/x'][:]
    y = h5_file['events/y'][:]
    t = h5_file['events/t'][:].astype(np.int64)
    p = h5_file['events/p'][:]
    p = ((p + 1) // 2).astype(np.uint8)
    cum_timestamps = np.cumsum([0] + new_timestamps)
    cum_timestamps += t_min
    num_frames = len(new_timestamps)
    with h5py.File(h5_output_path, 'w') as f_out:
        dset = f_out.create_dataset(
            '/data',
            shape=(num_frames, 2 * T, H, W),
            dtype='uint8',
            chunks=(1, 2 * T, H, W),
            **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
        )
        for i in tqdm(range(num_frames), desc="Processing Frames"): # Add tqdm here
            t_start = cum_timestamps[i]
            t_end = cum_timestamps[i + 1]
            mask = (t >= t_start) & (t < t_end)
            if not np.any(mask):
                continue
            x_slice = torch.from_numpy(x[mask].astype(np.int32)).to(device)
            y_slice = torch.from_numpy(y[mask].astype(np.int32)).to(device)
            t_slice = torch.from_numpy(t[mask].astype(np.int64)).to(device)
            p_slice = torch.from_numpy(p[mask]).to(device)
            rel_t = t_slice - t_start
            frame_dt = t_end - t_start
            bin_idx = ((rel_t % frame_dt) * T // frame_dt).long()
            bin_idx = torch.clamp(bin_idx, 0, T - 1)
            ch_idx = p_slice * T + bin_idx
            valid = (x_slice >= 0) & (x_slice < W) & (y_slice >= 0) & (y_slice < H)
            x_valid = x_slice[valid]
            y_valid = y_slice[valid]
            ch_valid = ch_idx[valid]
            tensor = torch.zeros(2 * T, H, W, dtype=torch.int32, device=device)
            for x_v, y_v, ch_v in zip(x_valid, y_valid, ch_valid):
                tensor[ch_v, y_v, x_v] += 1
            tensor = torch.clamp(tensor, max=255).to(torch.uint8).cpu().numpy()
            dset[i] = tensor
            
    return cum_timestamps[1:]

def convert_h5_resolution_in_memory(h5_input_path):
    with h5py.File(h5_input_path, 'r') as f_in:
        x = f_in['/events/x'][:]
        y = f_in['/events/y'][:]
        p = f_in['/events/p'][:]
        t = f_in['/events/t'][:]
        t_offset = f_in['/t_offset'][()]
        ms_to_idx = f_in['/ms_to_idx'][:]
    # Escalar Y de 480 a 360 con dithering para evitar bandas
    y_float = y * 3 / 4
    jitter = np.random.uniform(-0.5, 0.5, size=y.shape)
    y_scaled = np.floor(y_float + jitter).astype(np.int32)
    y_scaled = np.clip(y_scaled, 0, 359)
    return {
        'events': {
            'x': x.astype(np.uint16),
            'y': y_scaled.astype(np.uint16),
            'p': p.astype(np.uint8),
            't': t.astype(np.uint64),
        },
        't_offset': np.uint64(t_offset),
        'ms_to_idx': ms_to_idx.astype(np.uint64),
    }

def save_event_image(h5_file_path, output_path, num_events=50000, resolution=(640, 480)):
    """
    Extrae y guarda una imagen de eventos (sin plot) desde el centro del archivo .h5.
    Parámetros:
        h5_file_path (str): Ruta al archivo .h5
        output_path (str): Ruta donde se guardará la imagen
        num_events (int): Número de eventos a extraer
        resolution (tuple): Resolución del sensor (ancho, alto)
    """
    with h5py.File(h5_file_path, 'r') as f:
        x = f['/events/x'][:]
        y = f['/events/y'][:]
        p = f['/events/p'][:]
        t = f['/events/t'][:]
        t_offset = f['/t_offset'][()]
        t = t + t_offset
    total_events = len(x)
    start = total_events // 2 - num_events // 2
    end = start + num_events
    x = x[start:end]
    y = y[start:end]
    p = p[start:end]
    # Crear imagen en blanco
    w, h = resolution
    event_image = np.zeros((h, w), dtype=np.int16)
    for xi, yi, pi in zip(x, y, p):
        if 0 <= xi < w and 0 <= yi < h:
            event_image[yi, xi] += 1 if pi else -1
    # Crear figura sin bordes ni ejes
    fig = plt.figure(frameon=False)
    fig.set_size_inches(w / 100, h / 100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(event_image, cmap='bwr', vmin=-5, vmax=5)
    fig.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Imagen de eventos guardada en: {output_path}")

def scale_y_coordinate(labels, original_height, new_height):
    
    new_labels = labels.copy()
    scale_factor = new_height / original_height
    # Scale the 'y' coordinate
    # Ensure 'y' field is numeric before scaling, though <f4> in dtype implies it is.
    if np.issubdtype(new_labels['y'].dtype, np.number):
        scaled_y = new_labels['y'] * scale_factor
        new_labels['y'] = scaled_y.astype(np.int32) # Convert to int32
        new_labels['y'] = np.clip(new_labels['y'], 0, new_height - 1) # Clip to new resolution bounds
    else:
        print(f"Warning: 'y' field (dtype: {new_labels['y'].dtype}) is not numeric. Skipping scaling.")
        # Save the unmodified array if 'y' is not numeric to prevent errors
        return
    return new_labels

def load_labels(npy_input_path):
    """
    Carga las etiquetas desde un archivo .npy.
    Parámetros:
        npy_input_path (str): Ruta al archivo .npy
    Retorna:
        np.ndarray: Array de etiquetas cargado
    """
    try:
        labels = np.load(npy_input_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: Input file not found at {npy_input_path}")
        return
    except Exception as e:
        print(f"Error loading NPY file {npy_input_path}: {e}")
        return
    # Basic validation for the expected structured array format
    if not (labels.ndim == 1 and labels.dtype.fields and 'y' in labels.dtype.names):
        print(f"Error: Input NPY file at {npy_input_path} does not contain a 1D structured array "
              f"with a 'y' field.")
        print(f"Array shape: {labels.shape}, dtype: {labels.dtype}")
        
        return
    
    return labels

def change_class_id(labels):
    # acces to class_id of npy_input_path
    
    # Label with the new class_ids
    new_labels = labels.copy()
    # Remove rows where class_id is 1,3,4 or 7
    new_labels = new_labels[np.isin(new_labels['class_id'], [0,1,2])]
    # Change the class_id depending on the labels
    
    condition = (new_labels['class_id'] == 5) | (new_labels['class_id'] == 6)
    new_labels['class_id'] = np.where(condition, 1, new_labels['class_id'])
    return new_labels
    
def write_event_data_to_memory_h5(data_dict):
    buffer = io.BytesIO()
    with h5py.File(buffer, 'w') as f:
        events_group = f.create_group('events')
        events_group.create_dataset('x', data=data_dict['events']['x'])
        events_group.create_dataset('y', data=data_dict['events']['y'])
        events_group.create_dataset('p', data=data_dict['events']['p'])
        events_group.create_dataset('t', data=data_dict['events']['t'])
        f.create_dataset('t_offset', data=data_dict['t_offset'])
        f.create_dataset('ms_to_idx', data=data_dict['ms_to_idx'])
    buffer.seek(0)
    return buffer

def filter_zero_size_bboxes(labels):
    """
    Elimina etiquetas con bounding boxes que tienen ancho o alto igual a cero.
    Parámetros:
        labels (np.ndarray): Array de etiquetas con campos 'w' y 'h'
    Retorna:
        np.ndarray: Array de etiquetas filtradas
    """
    valid = (labels['w'] > 0) & (labels['h'] > 0)
    return labels[valid]

def scale_bounding_boxes(labels, scale_x=1.0, scale_y=1.0):
    labels = labels.copy()
    labels['x'] = (labels['x'] * scale_x).astype(np.int32)
    labels['y'] = (labels['y'] * scale_y).astype(np.int32)
    labels['w'] = np.maximum(1, (labels['w'] * scale_x).astype(np.int32))
    labels['h'] = np.maximum(1, (labels['h'] * scale_y).astype(np.int32))
    return labels

def save_event_label_overlay(
    events, labels, 
    H=360, W=640, 
    save_path='event_label_overlay.png',
    num_labels=20
):
    """
    Guarda una imagen de eventos con las etiquetas (bounding boxes) superpuestas.
    
    Parámetros:
        events (dict): Diccionario con arrays 'x', 'y' de eventos
        labels (np.ndarray): Estructura con campos 'x', 'y', 'w', 'h'
        H, W (int): Altura y anchura del frame
        save_path (str): Ruta donde guardar la imagen
        num_labels (int): Número de bounding boxes a visualizar
    """
    # Crear fondo de eventos
    frame = np.zeros((H, W), dtype=np.uint8)
    x = events['x']
    y = events['y']
    frame[y, x] = 255  # Simple binary frame
    # Crear figura
    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
    ax.imshow(frame, cmap='gray')
    for det in labels[:num_labels]:
        rect = plt.Rectangle(
            (det['x'], det['y']), det['w'], det['h'],
            edgecolor='red', facecolor='none', linewidth=1
        )
        ax.add_patch(rect)
    ax.axis('off')
    plt.tight_layout(pad=0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved overlay to {save_path}")

def process_and_save_event_tensor_sequence(
    h5_file,
    h5_output_path='output_tensor_sequence.h5',
    T=10,
    frame_durations=None,
    H=360,
    W=640,
    batch_size=10,
    verbose=False,
    align_t_min=None
):
    if align_t_min is None:
        raise ValueError("align_t_min must be provided to ensure event/label alignment.")
    if frame_durations is None:
        raise ValueError("frame_durations must be provided as a list or array of frame durations in µs.")
    
    frame_durations = np.array(frame_durations, dtype=np.int64)
    num_frames = len(frame_durations)

    # --- Debugging Event Timestamps ---
    # ... (código existente de depuración) ...
    # --- End Debugging Block ---

    # Adjust t_min using actual event timestamps
    t = h5_file['events/t'][:].astype(np.int64) # Cargar 't' solo una vez
    
    # Comprobación de alineación
    if align_t_min > t.max():
        raise ValueError(f"align_t_min ({align_t_min}) is after last event timestamp ({t.max()}) — no events will be processed.")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print(f"Using device: {device}")
    
    # Cargar todos los eventos necesarios una sola vez al inicio
    x_all = h5_file['events/x'][:]
    y_all = h5_file['events/y'][:]
    p_all = h5_file['events/p'][:]
    p_all = ((p_all + 1) // 2).astype(np.uint8) # Pre-procesar polaridad

    if verbose:
        print(f"Processing {num_frames} variable-duration frames")

    with h5py.File(h5_output_path, 'w') as f_out:
        dset = f_out.create_dataset(
            '/data',
            shape=(num_frames, 2 * T, H, W),
            dtype='uint8',
            chunks=(1, 2 * T, H, W),
            **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
        )
        
        # Calcular los tiempos de inicio y fin de cada frame
        frame_start_times = np.concatenate([[align_t_min], align_t_min + np.cumsum(frame_durations[:-1])])
        frame_end_times = align_t_min + np.cumsum(frame_durations)

        # Iterar por batches
        for batch_start in tqdm(range(0, num_frames, batch_size), desc="Processing Event Batches"):
            batch_end = min(batch_start + batch_size, num_frames)
            current_batch_size = batch_end - batch_start

            # Determinar el rango de tiempo total para el batch actual
            t_batch_start = frame_start_times[batch_start]
            t_batch_end = frame_end_times[batch_end - 1] # Fin del último frame en el batch

            # Filtrar todos los eventos que caen dentro del rango de tiempo de todo el batch
            mask_batch = (t >= t_batch_start) & (t < t_batch_end)
            
            # Si no hay eventos en este batch, llenar con ceros y continuar
            if not np.any(mask_batch):
                batch_tensor = torch.zeros((current_batch_size, 2 * T, H, W), dtype=torch.uint8, device=device)
                dset[batch_start:batch_end] = batch_tensor.cpu().numpy()
                if verbose:
                    print(f"Frames {batch_start}–{batch_end - 1} written (no events)")
                continue

            # Eventos relevantes para el batch
            x_batch = torch.from_numpy(x_all[mask_batch].astype(np.int32)).to(device)
            y_batch = torch.from_numpy(y_all[mask_batch].astype(np.int32)).to(device)
            t_batch = torch.from_numpy(t[mask_batch].astype(np.int64)).to(device)
            p_batch = torch.from_numpy(p_all[mask_batch]).to(device)

            # Encontrar a qué frame de salida pertenece cada evento
            # Usar searchsorted para encontrar el índice del frame de inicio más cercano para cada evento
            # Esto asigna cada evento a su frame correspondiente dentro del batch
            # Los frame_start_times_batch son los inicios de los frames DENTRO del batch actual
            frame_start_times_batch_torch = torch.from_numpy(frame_start_times[batch_start:batch_end]).to(device)
            
            # Buscar el índice del frame al que pertenece cada evento.
            # `right` asegura que un evento en `t_start` se asigne al bin `t_start`.
            # `torch.searchsorted` es clave aquí para vectorizar la asignación de frames.
            # El resultado 'frame_idx_in_batch' será un índice relativo dentro del batch (0 a current_batch_size-1)
            frame_idx_in_batch = torch.searchsorted(frame_start_times_batch_torch, t_batch, right=True) - 1
            
            # Filtrar eventos que caen fuera de los límites del batch o que no tienen un frame válido
            # Asegurarse de que frame_idx_in_batch esté dentro de los límites válidos
            valid_frame_idx_mask = (frame_idx_in_batch >= 0) & (frame_idx_in_batch < current_batch_size)
            
            x_batch_valid = x_batch[valid_frame_idx_mask]
            y_batch_valid = y_batch[valid_frame_idx_mask]
            t_batch_valid = t_batch[valid_frame_idx_mask]
            p_batch_valid = p_batch[valid_frame_idx_mask]
            frame_idx_in_batch_valid = frame_idx_in_batch[valid_frame_idx_mask]

            # Si después del filtrado no quedan eventos válidos para el batch, llenar con ceros
            if x_batch_valid.numel() == 0:
                batch_tensor = torch.zeros((current_batch_size, 2 * T, H, W), dtype=torch.uint8, device=device)
                dset[batch_start:batch_end] = batch_tensor.cpu().numpy()
                if verbose:
                    print(f"Frames {batch_start}–{batch_end - 1} written (no valid events after filtering)")
                continue
            
            # Obtener los tiempos de inicio para los frames a los que se asignaron los eventos
            # Esto se hace vectorialmente usando gather
            assigned_frame_start_times = frame_start_times_batch_torch[frame_idx_in_batch_valid]
            
            # Calcular el tiempo relativo dentro de su frame asignado
            rel_t_batch = t_batch_valid - assigned_frame_start_times
            
            # Obtener las duraciones de los frames asignados
            assigned_frame_durations = torch.from_numpy(frame_durations[batch_start:batch_end]).to(device)[frame_idx_in_batch_valid]

            # Calcular bin_idx (índice temporal dentro del frame)
            # Evitar división por cero si alguna duración es 0, aunque debería ser positiva
            bin_idx_batch = torch.div(rel_t_batch * T, assigned_frame_durations, rounding_mode='floor').long()
            bin_idx_batch = torch.clamp(bin_idx_batch, 0, T - 1)

            # Calcular ch_idx (canal de polaridad + bin temporal)
            ch_idx_batch = p_batch_valid * T + bin_idx_batch

            # Validar coordenadas de eventos
            valid_coords_mask = (x_batch_valid >= 0) & (x_batch_valid < W) & (y_batch_valid >= 0) & (y_batch_valid < H)
            
            x_final = x_batch_valid[valid_coords_mask]
            y_final = y_batch_valid[valid_coords_mask]
            ch_final = ch_idx_batch[valid_coords_mask]
            frame_idx_final = frame_idx_in_batch_valid[valid_coords_mask]

            # Crear el tensor 4D para el batch (batch_size, 2*T, H, W)
            batch_tensor_flat_size = current_batch_size * 2 * T * H * W
            batch_tensor_flat = torch.zeros(batch_tensor_flat_size, dtype=torch.int32, device=device)

            # Calcular los índices planos para la operación scatter_add_
            # flat_idx = frame_idx * (2*T*H*W) + ch_idx * (H*W) + y_coord * W + x_coord
            flat_idx_batch = (
                frame_idx_final * (2 * T * H * W) +
                ch_final * (H * W) +
                y_final * W +
                x_final
            ).long()

            # Realizar la operación scatter_add_ una sola vez para todo el batch
            batch_tensor_flat.scatter_add_(0, flat_idx_batch, torch.ones_like(flat_idx_batch, dtype=torch.int32))
            
            # Reshape y clamp a uint8
            batch_tensor = batch_tensor_flat.view(current_batch_size, 2 * T, H, W)
            batch_tensor = torch.clamp(batch_tensor, max=255).to(torch.uint8)
            
            # Guardar el batch completo en el dataset HDF5
            dset[batch_start:batch_end] = batch_tensor.cpu().numpy()
            if verbose:
                print(f"Frames {batch_start}–{batch_end - 1} written")
    
    print(f"\nSaved tensor sequence to: {h5_output_path}")
    return num_frames, align_t_min

# ----------------- PIPELINE -----------------------
interval = 50000  # Intervalo de tiempo entre frames en microsegundos
dt_in_ms = interval / 1000  # Convertir a milisegundos
T = 10  # Número de bins para el histograma
# --- Configuración de rutas y parámetros
dataset= 'dsec'
split = 'test'
input_path = 'data/' + dataset + '/' + split 
dest_path = 'data/' + dataset + '_proc' + '/' + split
sequences = [item for item in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, item))]
sequences = ["interlaken_00_b"] # Hardcoded for testing, remember to generalize or comment out for full run.

for sequence in sequences:
    print(f"Processing sequence: {sequence}")
    
    ev_rep_save_path = dest_path  + '/'  + sequence + '/' + 'event_representations_v2/' + 'stacked_histogram_dt='+ str(int(dt_in_ms)) + '_nbins=' + str(T) +'/'
    label_save_path = dest_path  + '/'  + sequence + '/' + 'labels_v2/'
    
    # Inputs paths
    h5_input_path = input_path + '/' + sequence + "/events/events.h5"
    npy_input_path = input_path + '_object_detections/' + sequence + "/object_detections/left/tracks.npy" 
    h5_output_path = ev_rep_save_path + 'event_representations_ds2_nearest.h5'
    # --- Ensure folders exist ---
    os.makedirs(ev_rep_save_path, exist_ok=True)
    os.makedirs(label_save_path, exist_ok=True)

    # ----------------- LABELS ------------------------
    # Cargar y procesar etiquetas antes del procesamiento de eventos
    og_labels = load_labels(npy_input_path)
    scaled_labels = scale_bounding_boxes(og_labels, scale_x=1.0, scale_y=360/480)
    updated_class_ids_labels = change_class_id(scaled_labels)
    non_zero_bbox_labels = filter_zero_size_bboxes(updated_class_ids_labels)
    real_frame_count = get_number_of_rgb_frames(sequence)
    unprocessed_labels = non_zero_bbox_labels.copy()
    

    # ----------------- LOAD FIXED TIMESTAMPS FROM TXT ------------------------
    timestamps_txt_path = input_path + '/' + sequence + "/timestamps.txt"
    print(f"Loading frame timestamps from: {timestamps_txt_path}")
    with open(timestamps_txt_path, 'r') as f:
        timestamp_lines = f.readlines()

    # --- START OF MODIFICATION TO USE RELATIVE TIMESTAMPS FOR FRAMES ---
    original_absolute_timestamps_from_txt = np.array([int(line.strip()) for line in timestamp_lines], dtype=np.int64)

    # Calculate frame durations based on the original timestamps
    new_timestamps = np.diff(original_absolute_timestamps_from_txt) # This is the array of durations (e.g., [50017, 50006, ...])

    # Create reconstructed_timestamps_np as accumulated differences starting from 0
    # This means the first frame starts at 0, and subsequent frames end at accumulated durations.
    reconstructed_timestamps_np = np.concatenate([[0], np.cumsum(new_timestamps)]) # <--- MODIFIED LINE
    
    # t_min_label will now be 0, satisfying the condition align_t_min <= t.max()
    t_min_label = reconstructed_timestamps_np[0] # This will now be 0. <--- MODIFIED LINE
    # --- END OF MODIFICATION ---


    # assign a detection to the nearest reconstructed_timestamp
    # This part will now map labels to frame indices based on the new relative time scale.
    label_timestamps = unprocessed_labels["t"]
    reconstructed_timestamps_np_for_mapping = np.array(reconstructed_timestamps_np) # Use the new relative timestamps for mapping
    obj_idx_2_rep_idx = []
    for label_ts in label_timestamps:
        # Find the index of the nearest timestamp in reconstructed_timestamps_np_for_mapping
        # You'll need to decide how to handle the original 'label_ts' (absolute)
        # when matching against 'reconstructed_timestamps_np_for_mapping' (relative from 0).
        # This part might need further adjustment if the original label_ts are absolute.
        # For now, let's assume you want to map original label_ts to the *closest original absolute frame timestamp*.
        # So, we revert to using original_absolute_timestamps_from_txt for mapping here:
        idx = np.searchsorted(original_absolute_timestamps_from_txt, label_ts) # <--- MODIFIED LINE for correct mapping
        
        if idx == 0:
            nearest_idx = 0
        elif idx == len(original_absolute_timestamps_from_txt): # <--- MODIFIED LINE
            nearest_idx = len(original_absolute_timestamps_from_txt) - 1 # <--- MODIFIED LINE
        else:
            # Check the two nearest timestamps from the original absolute list
            ts1 = original_absolute_timestamps_from_txt[idx - 1] # <--- MODIFIED LINE
            ts2 = original_absolute_timestamps_from_txt[idx] # <--- MODIFIED LINE

            if abs(label_ts - ts1) < abs(label_ts - ts2):
                nearest_idx = idx - 1
            else:
                nearest_idx = idx

        obj_idx_2_rep_idx.append(nearest_idx)
    # Convert the list to a numpy array
    obj_idx_2_rep_idx = np.array(obj_idx_2_rep_idx)

    last_labels = non_zero_bbox_labels
    last_labels = last_labels.copy()
    last_labels['t'] = (last_labels['t']).astype(np.int64)
    
    # ----------------- EVENTOS ------------------------
    # Convertir archivo a nueva resolución en memoria
    event_data = convert_h5_resolution_in_memory(h5_input_path)
    # Escribir archivo .h5 en RAM (BytesIO)
    h5_buffer = write_event_data_to_memory_h5(event_data)
    # Procesar directamente desde el archivo en RAM
    with h5py.File(h5_buffer, 'r') as h5_file:
        num_frames, t_min_processed = process_and_save_event_tensor_sequence(
            h5_file=h5_file,
            h5_output_path=h5_output_path,
            T=T,
            H=360, # <--- MODIFIED: Changed from 384 to 360 for consistency with event/label scaling
            W=640,
            frame_durations=new_timestamps, # These are still the durations
            batch_size=20,
            verbose=True,
            align_t_min=t_min_label # This is now 0
        )

    # ----------------- TIMESTAMPS Y LABELS ------------------------
    # Guardar timestamps generados by the representations (now relative from 0)
    timestamps_output_path = ev_rep_save_path + 'timestamps_us.npy'
    print(f"Saving timestamps to: {timestamps_output_path}")
    np.save(timestamps_output_path, reconstructed_timestamps_np) # Saves the new relative timestamps <--- MODIFIED

    # save object index to representation index mapping
    obj_idx_2_rep_idx_output_path = ev_rep_save_path + 'objframe_idx_2_repr_idx.npy'
    print(f"Saving object index to representation index mapping to: {obj_idx_2_rep_idx_output_path}")
    np.save(obj_idx_2_rep_idx_output_path, obj_idx_2_rep_idx)

    # labels_v2
    labels_output_path = label_save_path + 'labels.npz'
    print(f"Saving labels to: {labels_output_path}")
        # Guardar estructura compatible con RVT
    np.savez(os.path.join(label_save_path, 'labels.npz'),
             labels=last_labels,
             objframe_idx_2_label_idx=obj_idx_2_rep_idx)
    
    # save the timestamps belonging to the labels (these are still the original absolute ones)
    label_timestamps_output_path = label_save_path + 'timestamps_us.npy'
    print(f"Saving label timestamps to: {label_timestamps_output_path}")
    np.save(label_timestamps_output_path, non_zero_bbox_labels['t'])

    # check the number of frames in the h5 file
    print(f"Number of frames processed and saved: {num_frames}")



















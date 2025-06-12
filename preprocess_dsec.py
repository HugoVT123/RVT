import hdf5plugin  # Necesario si el archivo usa compresión especial
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
import os
import shutil
import io


# Change the size of the file


def convert_h5_resolution(h5_input_path, h5_output_path):

    """
    Lee un archivo .h5 de eventos, escala la resolución vertical de 480 a 360
    y guarda un nuevo archivo .h5 con la misma estructura y datos ajustados.

    Parámetros:
        h5_input_path (str): Ruta del archivo .h5 original
        h5_output_path (str): Ruta donde se guardará el nuevo archivo .h5
    """
    with h5py.File(h5_input_path, 'r') as f_in:
        # Leer datos de eventos
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

    # ✨ Escribir nuevo archivo .h5 con misma estructura
    with h5py.File(h5_output_path, 'w') as f_out:
        events_group = f_out.create_group('events')
        events_group.create_dataset('x', data=x, dtype='uint16')
        events_group.create_dataset('y', data=y_scaled, dtype='uint16')
        events_group.create_dataset('p', data=p, dtype='uint8')
        events_group.create_dataset('t', data=t, dtype='uint64')

        f_out.create_dataset('t_offset', data=t_offset, dtype='uint64')
        f_out.create_dataset('ms_to_idx', data=ms_to_idx, dtype='uint64')

    print(f"Nuevo archivo .h5 guardado en: {h5_output_path}")

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
    
    new_labels['class_id'] = np.where(new_labels['class_id'] == 2, 0, new_labels['class_id'])

    condition = (new_labels['class_id'] == 5) | (new_labels['class_id'] == 6)
    new_labels['class_id'] = np.where(condition, 1, new_labels['class_id'])

    return new_labels
    
def filter_confidence(labels, threshold=0.5):
    """
    Filtra las etiquetas basadas en un umbral de confianza.

    Parámetros:
        labels (np.ndarray): Array de etiquetas con campo 'confidence'
        threshold (float): Umbral de confianza para filtrar

    Retorna:
        np.ndarray: Array filtrado de etiquetas
    """
    return labels[labels['class_confidence'] >= threshold]
    
def process_and_save_event_tensor_sequence_gpu_batched(
    h5_input_path,
    h5_output_path='output_tensor_sequence.h5',
    T=10,
    dt=50000,
    H=360,
    W=640,
    max_frames=None,
    batch_size=10,
    verbose=False,
    align_t_min=None
):
    if align_t_min is None:
        raise ValueError("align_t_min must be provided to ensure event/label alignment.")
    t_min = align_t_min
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print(f"Using device: {device}")

    with h5py.File(h5_input_path, 'r') as f_in:
        x = f_in['events/x'][:]
        y = f_in['events/y'][:]
        t = f_in['events/t'][:].astype(np.int64)  
        p = f_in['events/p'][:]
        p = ((p + 1) // 2).astype(np.uint8)

        t_min, t_max = t.min(), t.max()
        total_duration = t_max - t_min
        num_frames = total_duration // dt

        if max_frames is not None:
            num_frames = min(num_frames, max_frames)

        if verbose:
            print(f"Processing {num_frames} frames from {t_min} to {t_max} µs")

        with h5py.File(h5_output_path, 'w') as f_out:
            dset = f_out.create_dataset(
                '/data',
                shape=(num_frames, 2 * T, H, W),
                dtype='uint8',
                chunks=(1, 2 * T, H, W),
                **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
            )

            for batch_start in range(0, num_frames, batch_size):
                batch_end = min(batch_start + batch_size, num_frames)
                actual_batch_size = batch_end - batch_start

                t_batch_start = t_min + batch_start * dt
                t_batch_end = t_min + batch_end * dt

                # Get only events in this batch window
                mask = (t >= t_batch_start) & (t < t_batch_end)
                if not np.any(mask):
                    continue

                x_slice = torch.from_numpy(x[mask].astype(np.int32)).to(device)
                y_slice = torch.from_numpy(y[mask].astype(np.int32)).to(device)
                t_slice = torch.from_numpy(t[mask].astype(np.int64)).to(device)  
                p_slice = torch.from_numpy(p[mask]).to(device)

                rel_t = t_slice - t_batch_start
                frame_idx = (rel_t // dt).long()
                bin_idx = ((rel_t % dt) * T // dt).long()
                bin_idx = torch.clamp(bin_idx, 0, T - 1)

                ch_idx = p_slice * T + bin_idx

                valid = (x_slice >= 0) & (x_slice < W) & (y_slice >= 0) & (y_slice < H)
                x_valid = x_slice[valid]
                y_valid = y_slice[valid]
                ch_valid = ch_idx[valid]
                f_valid = frame_idx[valid]

                flat_idx = (
                    f_valid * (2 * T * H * W)
                    + ch_valid * (H * W)
                    + y_valid * W
                    + x_valid
                ).long()

                tensor_flat = torch.zeros(actual_batch_size * 2 * T * H * W, dtype=torch.int32, device=device)
                tensor_flat.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.int32))
                tensor_batch = tensor_flat.view(actual_batch_size, 2 * T, H, W)
                tensor_batch = torch.clamp(tensor_batch, max=255).to(torch.uint8).cpu().numpy()

                dset[batch_start:batch_end] = tensor_batch

                if verbose:
                    print(f"Frames {batch_start}–{batch_end - 1} written")

    print(f"\n Saved tensor sequence to: {h5_output_path}")
    return num_frames,t_min

def generate_timestamps_us(num_frames, t_min, dt, save_path=''):
    
    timestamps_us = np.array([t_min + (i+1) * dt for i in range(num_frames)], dtype=np.int64)
    np.save(save_path +'timestamps_us.npy', timestamps_us)

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

def process_and_save_event_tensor_sequence_gpu_batched_from_file(
    h5_file,
    h5_output_path='output_tensor_sequence.h5',
    T=10,
    dt=50000,
    H=360,
    W=640,
    max_frames=None,
    batch_size=10,
    verbose=False,
    align_t_min=None
):
    if align_t_min is None:
        raise ValueError("align_t_min must be provided to ensure event/label alignment.")
    t_min = align_t_min
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print(f"Using device: {device}")

    # Leer desde el archivo abierto (h5_file en lugar de ruta)
    x = h5_file['events/x'][:]
    y = h5_file['events/y'][:]
    t = h5_file['events/t'][:].astype(np.int64)
    p = h5_file['events/p'][:]
    p = ((p + 1) // 2).astype(np.uint8)

    t_min, t_max = t.min(), t.max()
    total_duration = t_max - t_min
    num_frames = total_duration // dt

    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    if verbose:
        print(f"Processing {num_frames} frames from {t_min} to {t_max} µs")

    with h5py.File(h5_output_path, 'w') as f_out:
        dset = f_out.create_dataset(
            '/data',
            shape=(num_frames, 2 * T, H, W),
            dtype='uint8',
            chunks=(1, 2 * T, H, W),
            **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
        )

        for batch_start in range(0, num_frames, batch_size):
            batch_end = min(batch_start + batch_size, num_frames)
            actual_batch_size = batch_end - batch_start

            t_batch_start = t_min + batch_start * dt
            t_batch_end = t_min + batch_end * dt

            mask = (t >= t_batch_start) & (t < t_batch_end)
            if not np.any(mask):
                continue

            x_slice = torch.from_numpy(x[mask].astype(np.int32)).to(device)
            y_slice = torch.from_numpy(y[mask].astype(np.int32)).to(device)
            t_slice = torch.from_numpy(t[mask].astype(np.int64)).to(device)
            p_slice = torch.from_numpy(p[mask]).to(device)

            rel_t = t_slice - t_batch_start
            frame_idx = (rel_t // dt).long()
            bin_idx = ((rel_t % dt) * T // dt).long()
            bin_idx = torch.clamp(bin_idx, 0, T - 1)

            ch_idx = p_slice * T + bin_idx

            valid = (x_slice >= 0) & (x_slice < W) & (y_slice >= 0) & (y_slice < H)
            x_valid = x_slice[valid]
            y_valid = y_slice[valid]
            ch_valid = ch_idx[valid]
            f_valid = frame_idx[valid]

            flat_idx = (
                f_valid * (2 * T * H * W)
                + ch_valid * (H * W)
                + y_valid * W
                + x_valid
            ).long()

            tensor_flat = torch.zeros(actual_batch_size * 2 * T * H * W, dtype=torch.int32, device=device)
            tensor_flat.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.int32))
            tensor_batch = tensor_flat.view(actual_batch_size, 2 * T, H, W)
            tensor_batch = torch.clamp(tensor_batch, max=255).to(torch.uint8).cpu().numpy()

            dset[batch_start:batch_end] = tensor_batch

            if verbose:
                print(f"Frames {batch_start}–{batch_end - 1} written")

    print(f"\nSaved tensor sequence to: {h5_output_path}")
    return num_frames, t_min

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


for sequence in sequences:
    
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
    #filtered_labels = filter_confidence(non_zero_bbox_labels, threshold=0.01)


    last_labels = non_zero_bbox_labels
    t_min_label = last_labels['t'].min()
    print(f"t_min_label: {t_min_label} µs")
    print(f"Total labels: {len(last_labels)}")

    # ----------------- EVENTOS ------------------------
    # Convertir archivo a nueva resolución en memoria
    event_data = convert_h5_resolution_in_memory(h5_input_path)

    # Escribir archivo .h5 en RAM (BytesIO)
    h5_buffer = write_event_data_to_memory_h5(event_data)

    # Procesar directamente desde el archivo en RAM
    with h5py.File(h5_buffer, 'r') as h5_file:
        num_frames, t_min = process_and_save_event_tensor_sequence_gpu_batched_from_file(
            h5_file=h5_file,
            h5_output_path=h5_output_path,
            T=T,
            dt=interval,
            H=360,
            W=640,
            max_frames=None,
            batch_size=50,
            verbose=True,
            align_t_min=t_min_label
        )

    # ----------------- TIMESTAMPS Y LABELS ------------------------
    # Guardar timestamps generados por las representaciones
    generate_timestamps_us(num_frames, t_min_label, interval, save_path=ev_rep_save_path)

    # Agrupar etiquetas por timestamp
    labels_per_frame = defaultdict(list)
    for det in last_labels:
        labels_per_frame[det['t']].append(det)

    # Ordenar y agrupar
    sorted_ts = sorted(labels_per_frame.keys())
    frame_timestamps_us = np.array(sorted_ts, dtype=np.int64)
    labels_grouped = [np.array(labels_per_frame[t], dtype=last_labels.dtype) for t in frame_timestamps_us]

    # Aplanar todas las etiquetas
    flat_labels = np.concatenate(labels_grouped, axis=0)

    # Índices para reconstrucción
    objframe_idx_2_label_idx = np.zeros(len(labels_grouped), dtype=np.int64)
    count = 0
    for i, group in enumerate(labels_grouped):
        objframe_idx_2_label_idx[i] = count
        count += len(group)

    # Guardar estructura compatible con RVT
    np.savez(os.path.join(label_save_path, 'labels.npz'),
             labels=flat_labels,
             objframe_idx_2_label_idx=objframe_idx_2_label_idx)

    # Guardar timestamps de representaciones
    event_repr_timestamps = np.load(os.path.join(ev_rep_save_path, 'timestamps_us.npy'))
    np.save(os.path.join(label_save_path, 'event_repr_timestamps.npy'), event_repr_timestamps)

    # Alinear timestamps de labels a representaciones
    event_repr_timestamps = np.load(os.path.join(ev_rep_save_path, 'timestamps_us.npy'))
    objframe_idx_2_repr_idx = np.searchsorted(event_repr_timestamps, frame_timestamps_us, side='left')
    print(f"First 10 objframe_idx_2_repr_idx: {objframe_idx_2_repr_idx[:10]}")  # Debugging
    np.save(os.path.join(ev_rep_save_path, 'objframe_idx_2_repr_idx.npy'), objframe_idx_2_repr_idx)















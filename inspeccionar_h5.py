import hdf5plugin
import h5py
import numpy as np
import torch

def inspeccionar_h5(ruta_archivo):
    """
    Abre un archivo H5, lista su contenido (datasets y grupos),
    e imprime la forma (shape) de cada dataset.
    Opcionalmente, imprime una pequeña muestra de los datos.
    """
    try:
        with h5py.File(ruta_archivo, 'r') as hf:
            print(f"Inspeccionando el archivo: {ruta_archivo}")
            print("-----------------------------------------")

            def recorrer_objetos(nombre, objeto):
                if isinstance(objeto, h5py.Dataset):
                    print(f"Dataset: {nombre}")
                    print(f"  Shape: {objeto.shape}")
                    print(f"  Tipo de datos (dtype): {objeto.dtype}")
                    # Opcional: Imprimir una pequeña muestra de los datos
                    # Ten cuidado con datasets muy grandes.
                    try:
                        if objeto.size > 0: # Solo si no está vacío
                            # Imprime las primeras filas/elementos, ajusta según la dimensionalidad
                            if objeto.ndim == 1:
                                print(f"  Primeros elementos (hasta 5): {objeto[:5]}")
                            elif objeto.ndim > 1:
                                print(f"  Primeras filas/elementos (hasta 5x...): {objeto[:min(5, objeto.shape[0])]}")
                            else: # Escalar
                                print(f"  Valor: {objeto[()]}")
                        else:
                            print("  Dataset está vacío.")
                    except Exception as e:
                        print(f"  No se pudo imprimir una muestra de los datos: {e}")
                    print("---")
                elif isinstance(objeto, h5py.Group):
                    print(f"Grupo: {nombre}")
                    print("---")
                # No es necesario hacer nada más para los grupos aquí,
                # hf.visititems se encargará de recorrerlos.

            hf.visititems(recorrer_objetos)

    except FileNotFoundError:
        print(f"Error: El archivo '{ruta_archivo}' no fue encontrado.")
    except Exception as e:
        print(f"Ocurrió un error al abrir o procesar el archivo H5: {e}")



def process_and_save_event_tensor_sequence_gpu_batched(
    h5_input_path,
    h5_output_path='output_tensor_sequence.h5',
    T=10,
    dt=50000,
    H=360,
    W=640,
    max_frames=None,
    batch_size=10,
    verbose=False
):
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

                #x_slice = torch.from_numpy(x[mask]).to(device)
                #y_slice = torch.from_numpy(y[mask]).to(device)
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


# --- Uso del script ---
# Reemplaza 'tu_archivo.h5' con la ruta real a tu archivo.
ruta_del_archivo_h5 = 'data/gen4_proc/train/moorea_2019-02-15_001_td_183500000_243500000/event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations_ds2_nearest.h5'  # Cambia esto por tu ruta real

ruta_del_archivo_h5_2 = 'data/gen4/moorea_2019-06-11_test02_000_3111500000_3171500000_td.h5'

tensor_seq = process_and_save_event_tensor_sequence_gpu_batched(h5_input_path='data/dummy_dsec/test/thun_00_a_td.h5',h5_output_path='DSEC_output_tensor_sequence_GPU.h5',T=10,dt=50000,H=360,W=640,max_frames=None, verbose=True)

#print(tensor_seq.shape)  # (100, 20, 360, 640)

#inspeccionar_h5('output_tensor_sequence_2.h5')
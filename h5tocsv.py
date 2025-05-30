import hdf5plugin  # Necesario si el archivo usa compresión especial
import h5py
import pandas as pd

def convert_h5_to_csv(h5_file_path, csv_output_path):
    """
    Convierte un archivo .h5 de eventos del dataset DSEC a formato CSV.
    
    Parámetros:
        h5_file_path (str): Ruta del archivo .h5
        csv_output_path (str): Ruta donde se guardará el archivo .csv
    """
    with h5py.File(h5_file_path, 'r') as f:
        x = f['/events/x'][:100]
        y = f['/events/y'][:100]
        p = f['/events/p'][:100]  # 0: negativa, 1: positiva
        t = f['/events/t'][:100]
        #t_offset = f['/t_offset'][()]
        #t = t + t_offset  # Ajustar tiempo

        # Crear un DataFrame
        df = pd.DataFrame({
            'timestamp_us': t,
            'x': x,
            'y': y,
            'polarity': p
        })

        # Guardar a CSV
        df.to_csv(csv_output_path, index=False)
        print(f"CSV guardado en: {csv_output_path}")

def convert_proc_h5_to_csv(h5_file_path, csv_output_path):
    """
    Convierte un archivo .h5 de eventos procesados del dataset DSEC a formato CSV.
    
    Parámetros:
        h5_file_path (str): Ruta del archivo .h5
        csv_output_path (str): Ruta donde se guardará el archivo .csv
    """
    with h5py.File(h5_file_path, 'r') as f:
        data = f['data'][:10]

        print(f"Datos cargados: {data}")
        

    


# Ejemplo de uso
dsec_proc_h5_path = 'thun_td_scaled.h5'          # Cambia esto por tu ruta
dsec_h5_path = 'data/dummy_dsec/test/thun_00_a_td.h5'          # Cambia esto por tu ruta

gen4_proc_og_h5_path = 'moorea_2019-02-15_001_td_183500000_243500000/event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations_ds2_nearest.h5'          # Cambia esto por tu ruta
gen4_proc_h5_generated_path = 'output_tensor_sequence_GPU.h5'          # Cambia esto por tu ruta

convert_h5_to_csv('data/gen4/moorea_2019-06-11_test02_000_3111500000_3171500000_td.h5', "gen4_100.csv")  # Cambia el nombre del archivo CSV si es necesario






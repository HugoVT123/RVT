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
        x = f['/events/x'][:1000]
        y = f['/events/y'][:1000]
        p = f['/events/p'][:1000]  # 0: negativa, 1: positiva
        t = f['/events/t'][:]

        # get rid of repeated timestamps
        t = pd.Series(t).drop_duplicates().values

        # get the last timestamp
        t_max = t[-1] if len(t) > 0 else 0
        

        print(t_max)

        #t_offset = f['/t_offset'][()]
        #t = t + t_offset  # Ajustar tiempo

        # Crear un DataFrame
        """ df = pd.DataFrame({
            'timestamp_us': t,
        }) """

        # Guardar a CSV
        #df.to_csv(csv_output_path, index=False)
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
dsec_rooth_path = 'data/dsec/test'          
sequence =  "interlaken_00_b" 

sequence_path = f"{dsec_rooth_path}/{sequence}/events/events.h5"


convert_h5_to_csv(sequence_path, f"{sequence}.csv")  # Cambia el nombre del archivo CSV si es necesario






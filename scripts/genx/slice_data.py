import sys
import h5py
from pathlib import Path
sys.path.append('../..')
from utils.preprocessing import _blosc_opts



in_path = Path("../../data/original_dsec/zurich_city_14_a_td.h5")
out_path = Path("../../data/dummy_dsec/test/chunked_zurich_city_14_a_td.h5")
chunk_size = 2**16  # or 4096, 16384, etc.

with h5py.File(in_path, 'r') as fin, h5py.File(out_path, 'w') as fout:
    events_in = fin['events']
    events_out = fout.create_group('events')
    print("length of events_in", len(events_in))
    print("length of events_out", len(events_out))
    print("Data examples from input:")
    print(list(events_in.keys()))
        

    for key in events_in:
        print("key", key)
        data = events_in[key][:]
        print(events_in[key][0:10])
        print(f"Processing dataset: {key}, shape: {data.shape}, dtype: {data.dtype}")
        events_out.require_dataset(
            key,
            data=data,
            dtype=data.dtype,
            shape=data.shape,
            chunks=(chunk_size,),
            **_blosc_opts(complevel=1, shuffle='byte')
        )

print(f"Chunked file saved to {out_path}")
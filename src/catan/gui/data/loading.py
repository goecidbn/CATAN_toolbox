# from pathlib import Path
# import h5py
# import numpy as np
# from scipy import sparse
# from typing import List, Optional


# def load_from_file(path: Path, fields: Optional[str | List[str]] = None) -> dict:

#     ext = Path(path).suffix.lower()
#     if ext in (".h5", ".hdf5"):
#         try:
#             return _load_fields_from_h5(path, fields, subpath="/estimates")
#         except:
#             return _load_fields_from_h5(path, fields)
#     elif ext == ".mat":
#         data = loadmat(path)
#         if fields is None:
#             return data
#         else:
#             return {field: data[field] for field in fields if field in data}
#     else:
#         raise ValueError(f"Unsupported file extension: {ext}")


# def _load_fields_from_h5(
#     path: Path, fields: Optional[str | List[str]], subpath: Optional[str] = None
# ) -> dict:
#     if fields is not None and not isinstance(fields, list):
#         fields = [fields]

#     out = {}
#     with h5py.File(path, "r") as f:
#         # print(np.array(f[subpath + "/C"]))
#         if subpath:
#             f = f[subpath]

#         if fields is None:
#             fields = list(f.keys())

#         print(f"Available fields in {path}: {list(f.keys())}")
#         # adjust path if needed for your CaImAn export (e.g. "/estimates/Cn")
#         for key in fields:
#             if key not in f.keys():
#                 print(f"Field {key} not found in {path}")
#                 continue
#             obj = f[key]
#             # print(f"loading {key} from {path}", obj)
#             if isinstance(obj, h5py.Group):
#                 ## --- Sparse matrix group (CaImAn style) ---
#                 if all(d in obj for d in ("indptr", "indices", "data", "shape")):
#                     data = obj["data"]
#                     indices = obj["indices"]
#                     indptr = obj["indptr"]
#                     shape = obj["shape"]
#                     out[key] = sparse.csc_matrix(
#                         (data[:], indices[:], indptr[:]), shape[:]
#                     )
#                 else:
#                     print("What to do???")
#                     # # For "normal" groups: list immediate datasets inside that group
#                     # for subkey, subobj in obj.items():
#                     #     if isinstance(subobj, h5py.Dataset):
#                     #         name = f"{field}/{subkey}"
#                     #         shape_str = str(subobj.shape)
#                     #         dtype_str = str(subobj.dtype)
#                     #         return (name, shape_str, dtype_str)

#             # --- Top-level dataset ---
#             elif isinstance(obj, h5py.Dataset):
#                 out[key] = np.array(f[key])
#         return out  # if len(out) > 1 else out[fields[0]]


# # def load_background(path: str, field: str) -> np.ndarray:
# #     """Load background image from HDF5 file."""
# #     if not path or not field:
# #         raise ValueError("Please specify a valid background file and field.")
# #     background_data = _load_field_from_h5(Path(path), field)
# #     return background_data

# from scipy import io as spio


# def loadmat(filename):
#     """
#     this function should be called instead of direct spio.loadmat
#     as it cures the problem of not properly recovering python dictionaries
#     from mat files. It calls the function check keys to cure all entries
#     which are still mat-objects
#     """
#     data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)

#     ### get rid of some unnecessary entries
#     for key in ["__header__", "__version__", "__globals__"]:
#         del data[key]

#     return _check_keys(data)


# def _check_keys(dict):
#     """
#     checks if entries in dictionary are mat-objects. If yes
#     todict is called to change them to nested dictionaries
#     """
#     for key in dict:
#         if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
#             dict[key] = _todict(dict[key])
#     return dict


# def _todict(matobj):
#     """
#     A recursive function which constructs from matobjects nested dictionaries
#     """
#     dict = {}
#     for strg in matobj._fieldnames:
#         elem = matobj.__dict__[strg]
#         if isinstance(elem, spio.matlab.mio5_params.mat_struct):
#             dict[strg] = _todict(elem)
#         else:
#             dict[strg] = elem
#     return dict

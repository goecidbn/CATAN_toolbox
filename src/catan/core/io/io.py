from __future__ import annotations

from typing import Any, Optional, List

import logging
from catan.core.structures.load_config import FieldSpec
import numpy as np
import pickle, h5py
from pathlib import Path, PurePosixPath
from scipy import sparse
from scipy import io as spio

ATTRIBUTES_KEY = "__attrs__"



# def load_data(loadPath, fields=None, **kwargs):

#     ext = Path(loadPath).suffix.lower()
#     if ext == ".hdf5":
#         ld = load_data_from_hdf5(loadPath, fields, **kwargs)
#     elif ext == ".pkl":
#         with open(loadPath, "rb") as f:
#             ld = pickle.load(f)
#     elif ext == ".mat":
#         ld = loadmat(loadPath)
#     else:
#         assert False, "File extension not yet implemented for loading data!"
#     return ld


# def save_data(data: dict, filename: str, **kwargs) -> None:

#     ext = Path(filename).suffix.lower()
#     if ext == ".hdf5":
#         # print("Saving data to hdf5 file...")
#         save_data_to_hdf5(data, filename, **kwargs)
#     elif ext == ".pkl":
#         with open(filename, "wb") as f:
#             pickle.dump(data, f)
#     elif ext == ".mat":
#         sv_data = {}
#         for key in data:
#             sv_data[str(key)] = data[key]
#             if isinstance(data[key], dict):
#                 for keyy in data[key]:
#                     if data[key][keyy] is None:
#                         sv_data[str(key)][keyy] = np.array([])
#         spio.savemat(filename, sv_data)
#     else:
#         assert False, "File extension not yet implemented for saving data!"

#     print(f"Data saved to {filename}.")






### -------------------------------------------------------- ###
### --------------------- save helpers --------------------- ###
### -------------------------------------------------------- ###


# def save_data_to_hdf5(dic: dict, filename: str, subdir: str = "/") -> None:
#     """Save dictionary to hdf5 file
#     Args:
#         dic: dictionary
#             input (possibly nested) dictionary
#         filename: str
#             file name to save the dictionary to (in hdf5 format for now)
#     """
#     # From https://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py

#     with h5py.File(filename, "w") as h5file:
#         recursively_save_dict_contents_to_group(h5file, subdir, dic)


# def recursively_save_dict_contents_to_group(
#     h5file: h5py.File, path: str, dic: dict, logLevel=logging.WARNING
# ) -> None:
#     """
#     Args:
#         h5file: hdf5 object
#             hdf5 file where to store the dictionary
#         path: str
#             path within the hdf5 file structure
#         dic: dictionary
#             dictionary to save
#     """
#     logger = logging.getLogger("caiman")
#     logger.setLevel(logLevel)
#     # argument type checking
#     if not isinstance(dic, dict):
#         raise ValueError("must provide a dictionary")

#     if not isinstance(path, str):
#         raise ValueError("path must be a string")

#     if not isinstance(h5file, h5py.File):
#         raise ValueError("must be an open h5py file")

#     # save items to the hdf5 file
#     for key, item in dic.items():
#         key = str(key)

#         if isinstance(item, (list, tuple)):
#             # print(f"{key} is list")
#             if len(item) > 0 and all(isinstance(elem, (Path, str)) for elem in item):
#                 # print(f"save {key}")
#                 # item = np.string_(item)
#                 item = np.bytes_(item)
#                 # pass

#             else:
#                 item = np.array(item)
#         if not isinstance(key, str):
#             raise ValueError("dict keys must be strings to save to hdf5")
#         # save strings, numpy.int64, numpy.int32, and numpy.float64 types
#         if isinstance(item, str):
#             logger.debug(f"Saving string {key}: {item}")
#             if path not in h5file:
#                 h5file.create_group(path)
#             h5file[path].attrs[key] = item
#         elif isinstance(item, (float, int)) or isinstance(
#             item, (np.integer, np.floating)
#         ):
#             # TODO In the future we may store all scalars, including these, as attributes too, although strings suffer the most from being stored as datasets
#             h5file[path + key] = item
#             logger.debug(f"Saving numeric {path + key}")
#             if not h5file[path + key][()] == item:
#                 raise ValueError(
#                     f"Error (v {h5py.__version__}) while saving numeric {path + key}: assigned value {h5file[path + key][()]} does not match intended value {item}"
#                 )
#         # save numpy arrays
#         elif isinstance(item, np.ndarray):
#             logger.debug(f"Saving {key}")
#             try:
#                 h5file[path + key] = item
#             except:
#                 item = np.array(item).astype("|S32")
#                 h5file[path + key] = item
#             if not np.array_equal(
#                 h5file[path + key][()], item, equal_nan=item.dtype.kind == "f"
#             ):  # just using True gives "ufunc 'isnan' not supported for the input types"
#                 raise ValueError(
#                     f"Error while saving ndarray {key} of dtype {item.dtype}"
#                 )
#         # save dictionaries
#         elif isinstance(item, dict):
#             recursively_save_dict_contents_to_group(h5file, path + key + "/", item)
#         elif "sparse" in str(type(item)):
#             logger.info(f"{key} is sparse ****")
#             h5file[path + key + "/data"] = item.tocsc().data
#             h5file[path + key + "/indptr"] = item.tocsc().indptr
#             h5file[path + key + "/indices"] = item.tocsc().indices
#             h5file[path + key + "/shape"] = item.tocsc().shape
#         # other types cannot be saved and will result in an error
#         elif item is None or key == "dview":
#             h5file[path + key] = "NoneType"
#         elif key in [
#             "dims",
#             "medw",
#             "sigma_smooth_snmf",
#             "dxy",
#             "max_shifts",
#             "strides",
#             "overlaps",
#             "gSig",
#         ]:
#             logger.info(f"{key} is a tuple ****")
#             h5file[path + key] = np.array(item)
#         elif type(item).__name__ in [
#             "CNMFParams",
#             "Estimates",
#             "session_data",
#             "remap_data",
#         ]:  #  parameter object
#             recursively_save_dict_contents_to_group(
#                 h5file, path + key + "/", item.__dict__
#             )
#         else:

#             raise ValueError(f"Cannot save {type(item)} type for key '{key}'.")


### =============================================================== ###
### ====================== HELPER FUNCTIONS ======================= ###
### =============================================================== ###


def fix_suffix(suffix):
    if suffix:
        if not suffix.startswith("_"):
            suffix = "_" + suffix
    return suffix

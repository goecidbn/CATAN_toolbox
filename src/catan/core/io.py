from typing import Any, Optional, List

import logging
import numpy as np
import pickle, h5py
from pathlib import Path
from scipy import sparse
from scipy import io as spio


def load_data(loadPath, fields=None, **kwargs):

    ext = Path(loadPath).suffix.lower()
    if ext == ".hdf5":
        ld = load_data_from_hdf5(loadPath, fields, **kwargs)
    elif ext == ".pkl":
        with open(loadPath, "rb") as f:
            ld = pickle.load(f)
    elif ext == ".mat":
        ld = loadmat(loadPath)
    else:
        assert False, "File extension not yet implemented for loading data!"
    return ld


def save_data(data: dict, filename: str, **kwargs) -> None:

    ext = Path(filename).suffix.lower()
    if ext == ".hdf5":
        # print("Saving data to hdf5 file...")
        save_data_to_hdf5(data, filename, **kwargs)
    elif ext == ".pkl":
        with open(filename, "wb") as f:
            pickle.dump(data, f)
    elif ext == ".mat":
        sv_data = {}
        for key in data:
            sv_data[str(key)] = data[key]
            if isinstance(data[key], dict):
                for keyy in data[key]:
                    if data[key][keyy] is None:
                        sv_data[str(key)][keyy] = np.array([])
        spio.savemat(filename, sv_data)
    else:
        assert False, "File extension not yet implemented for saving data!"

    print(f"Data saved to {filename}.")


### -------------------------------------------------------- ###
### --------------------- load helpers --------------------- ###
### -------------------------------------------------------- ###


def load_data_from_hdf5(
    path: str | Path,
    fields: Optional[str | List[str]] = None,
    subpath: str = "/estimates",
) -> dict:
    # function kinda from CaImAn (except enabled partial loading)
    if not Path(path).exists() or not Path(path).parts[-1].lower().endswith(
        (".h5", ".hdf5")
    ):
        raise ValueError(f"File {path} does not exist or is not an HDF5 file.")

    if fields is not None and not isinstance(fields, list):
        fields = [fields]
        flat_output = True
    else:
        flat_output = False

    out = {}
    with h5py.File(path, "r") as f:

        if subpath:
            f = f[subpath]

        for akey, aitem in f.attrs.items():
            out[akey] = aitem

        if not isinstance(f, h5py.Group):
            raise ValueError(f"Subpath {subpath} does not point to a Group in {path}")

        if fields is None:
            fields = list(f.keys())

        # print(f"Available fields in {path}: {list(f.keys())}")
        for key in fields:
            if key not in f.keys():
                print(f"Field {key} not found in {path}")
                continue
            obj = f[key]
            if isinstance(obj, h5py.Group):
                # print(f"is group {key}")
                ## --- Sparse matrix group (CaImAn style) ---
                if all(d in obj for d in ("indptr", "indices", "data", "shape")):
                    out[key] = read_sparse_matrix(obj)
                    # data = obj["data"]
                    # indices = obj["indices"]
                    # indptr = obj["indptr"]
                    # shape = tuple(obj["shape"][:])
                    # out[key] = sparse.csc_matrix(
                    #     (data[:], indices[:], indptr[:]), shape
                    # )
                else:
                    ### recursivity is not entirely clean (reloading of complete file, ...)
                    ### but works for now
                    out[key] = load_data_from_hdf5(
                        path, None, subpath=subpath + "/" + key
                    )
                    # raise ValueError(
                    #     f"Group {key} in {path} is not a recognized sparse matrix format."
                    # )

            # --- Top-level dataset ---
            elif isinstance(obj, h5py.Dataset):
                out[key] = f[key][:] if f[key].ndim > 0 else f[key][()]
                if isinstance(out[key], bytes):
                    out[key] = decode_hdf5_value(out[key])

    if flat_output and len(out) == 1:
        return next(iter(out.values()))
    return out


def decode_hdf5_value(value):
    if value == b"NoneType":
        return None

    if isinstance(value, bytes):
        return value.decode()

    return value


def loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)

    ### get rid of some unnecessary entries
    for key in ["__header__", "__version__", "__globals__"]:
        del data[key]

    return _check_keys(data)


def _check_keys(dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


### -------------------------------------------------------- ###
### --------------------- save helpers --------------------- ###
### -------------------------------------------------------- ###


def save_data_to_hdf5(dic: dict, filename: str, subdir: str = "/") -> None:
    """Save dictionary to hdf5 file
    Args:
        dic: dictionary
            input (possibly nested) dictionary
        filename: str
            file name to save the dictionary to (in hdf5 format for now)
    """
    # From https://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py

    with h5py.File(filename, "w") as h5file:
        recursively_save_dict_contents_to_group(h5file, subdir, dic)


def recursively_save_dict_contents_to_group(
    h5file: h5py.File, path: str, dic: dict, logLevel=logging.WARNING
) -> None:
    """
    Args:
        h5file: hdf5 object
            hdf5 file where to store the dictionary
        path: str
            path within the hdf5 file structure
        dic: dictionary
            dictionary to save
    """
    logger = logging.getLogger("caiman")
    logger.setLevel(logLevel)
    # argument type checking
    if not isinstance(dic, dict):
        raise ValueError("must provide a dictionary")

    if not isinstance(path, str):
        raise ValueError("path must be a string")

    if not isinstance(h5file, h5py._hl.files.File):
        raise ValueError("must be an open h5py file")

    # save items to the hdf5 file
    for key, item in dic.items():
        key = str(key)

        if isinstance(item, (list, tuple)):
            # print(f"{key} is list")
            if len(item) > 0 and all(isinstance(elem, (Path, str)) for elem in item):
                # print(f"save {key}")
                # item = np.string_(item)
                item = np.bytes_(item)
                # pass

            else:
                item = np.array(item)
        if not isinstance(key, str):
            raise ValueError("dict keys must be strings to save to hdf5")
        # save strings, numpy.int64, numpy.int32, and numpy.float64 types
        if isinstance(item, str):
            logger.debug(f"Saving string {key}: {item}")
            if path not in h5file:
                h5file.create_group(path)
            h5file[path].attrs[key] = item
        elif isinstance(item, (float, int)) or isinstance(
            item, (np.integer, np.floating)
        ):
            # TODO In the future we may store all scalars, including these, as attributes too, although strings suffer the most from being stored as datasets
            h5file[path + key] = item
            logger.debug(f"Saving numeric {path + key}")
            if not h5file[path + key][()] == item:
                raise ValueError(
                    f"Error (v {h5py.__version__}) while saving numeric {path + key}: assigned value {h5file[path + key][()]} does not match intended value {item}"
                )
        # save numpy arrays
        elif isinstance(item, np.ndarray):
            logger.debug(f"Saving {key}")
            try:
                h5file[path + key] = item
            except:
                item = np.array(item).astype("|S32")
                h5file[path + key] = item
            if not np.array_equal(
                h5file[path + key][()], item, equal_nan=item.dtype.kind == "f"
            ):  # just using True gives "ufunc 'isnan' not supported for the input types"
                raise ValueError(
                    f"Error while saving ndarray {key} of dtype {item.dtype}"
                )
        # save dictionaries
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + "/", item)
        elif "sparse" in str(type(item)):
            logger.info(f"{key} is sparse ****")
            h5file[path + key + "/data"] = item.tocsc().data
            h5file[path + key + "/indptr"] = item.tocsc().indptr
            h5file[path + key + "/indices"] = item.tocsc().indices
            h5file[path + key + "/shape"] = item.tocsc().shape
        # other types cannot be saved and will result in an error
        elif item is None or key == "dview":
            h5file[path + key] = "NoneType"
        elif key in [
            "dims",
            "medw",
            "sigma_smooth_snmf",
            "dxy",
            "max_shifts",
            "strides",
            "overlaps",
            "gSig",
        ]:
            logger.info(f"{key} is a tuple ****")
            h5file[path + key] = np.array(item)
        elif type(item).__name__ in [
            "CNMFParams",
            "Estimates",
            "session_data",
            "remap_data",
        ]:  #  parameter object
            recursively_save_dict_contents_to_group(
                h5file, path + key + "/", item.__dict__
            )
        else:

            raise ValueError(f"Cannot save {type(item)} type for key '{key}'.")


### =============================================================== ###
### ====================== HELPER FUNCTIONS ======================= ###
### =============================================================== ###


def write_sparse_matrix(
    group: h5py.Group,
    matrix: sparse.spmatrix,
) -> None:
    """Write a SciPy sparse matrix into an HDF5 group."""
    matrix = sparse.csc_matrix(matrix)

    group.attrs["format"] = "csc"
    group.attrs["shape"] = matrix.shape

    group.create_dataset("data", data=matrix.data)
    group.create_dataset("indices", data=matrix.indices)
    group.create_dataset("indptr", data=matrix.indptr)


def read_sparse_matrix(group: h5py.Group) -> sparse.csc_matrix:
    """Read a CSC matrix from an HDF5 group."""
    matrix_format = group.attrs.get("format", "csc")

    if isinstance(matrix_format, bytes):
        matrix_format = matrix_format.decode("utf-8")

    if matrix_format != "csc":
        raise ValueError(f"Unsupported sparse matrix format: {matrix_format!r}")

    if "shape" in group.keys():
        # ensure consistency with CaImAn style saving
        shape = tuple(group["shape"][()])
    else:
        shape = tuple(int(value) for value in group.attrs["shape"])

    return sparse.csc_matrix(
        (
            group["data"][()],
            group["indices"][()],
            group["indptr"][()],
        ),
        shape=shape,
    )


def write_optional_array(
    group: h5py.Group,
    name: str,
    value: np.ndarray | None,
    **dataset_kwargs: Any,
) -> None:
    if value is not None:
        group.create_dataset(name, data=value, **dataset_kwargs)


def read_optional_array(
    group: h5py.Group,
    name: str,
) -> np.ndarray | None:
    if name not in group:
        return None

    return group[name][()]


def write_optional_attr(
    group: h5py.Group,
    name: str,
    value: Any | None,
) -> None:
    if value is not None:
        group.attrs[name] = value


def read_optional_attr(
    group: h5py.Group,
    name: str,
    default: Any = None,
) -> Any:
    return group.attrs[name] if name in group.attrs else default

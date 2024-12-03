import os
import numpy as np


def random_init_grid(grid_size: int, 
                     q: float = 0.3701, seed=None) -> np.ndarray:
    if seed:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    assert 0 <= q <= 1
    return rng.choice([0, 1], p=[1-q, q], size=[grid_size, grid_size])


def random_patch(grid_size: int, patch_size: int, patch_top_left: tuple[int],
                 q: float = 0.3701, seed=None) -> np.ndarray:
    if seed:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    assert 0 <= q <= 1
    grid = np.zeros((grid_size, grid_size))
    patch = rng.choice([0, 1], p=[1-q, q], size=[patch_size, patch_size])
    i, j = patch_top_left
    assert i + patch_size <= grid_size and j + patch_size <= grid_size
    grid[i:i+patch_size, j:j+patch_size] = patch
    return grid


def save_data(data, param=None, header="", base_path=None, prefix=None,
              sub_path=None, overwrite_protection=True):
    # data: one or more numpy arrays
    # param: a value to be used to name the file appropriately
    # header: an optional string to be written at the beginning of the file
    # base_path: an optional string indicating the base directory where the file will be stored
    # prefix: an optional string to be used as the file name prefix
    # sub_path: an optional string indicating a subdirectory within the base path

    # check if data is a single array or a sequence of arrays
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy array")

    # check if the arrays have the same shape
    shape = data[0].shape
    for arr in data[1:]:
        if arr.shape != shape:
            raise ValueError("all arrays must have the same shape")

    # check if the header is a string
    if not isinstance(header, str):
        raise TypeError("header must be a string")

    # check if the base path is None, a valid directory, or an invalid value
    if base_path is None:
        base_path = os.getcwd() # use the current working directory
    elif not isinstance(base_path, str):
        raise TypeError("base path must be a string or None")

    # check if the prefix is a string
    if prefix is not None and not isinstance(prefix, str):
        raise TypeError("prefix must be a string")

    # check if the sub path is a string or None
    if sub_path is not None and not isinstance(sub_path, str):
        raise TypeError("sub path must be a string or None")
    
    if param is None and prefix is None:
        raise ValueError("provide file name clues")

    # format the parameter value as a valid file name
    if param is not None:
        param = "%.3f"%param # keep four decimal places and strip trailing zeros and dot
    else:
        param = ""
    
    if prefix is None:
        prefix = ""

    # construct the full path by joining the base path, sub path, and file name
    file_name = prefix + param + ".dat" # use the parameter value to name the file
    if sub_path is not None:
        full_path = os.path.join(base_path, sub_path, file_name) # append the sub path if given
    else:
        full_path = os.path.join(base_path, file_name) # otherwise, use only the base path
    
    if os.path.exists(full_path):
        answer = input(f"The file {file_name} already exists. Do you want to overwrite it? (Y/N) ")
        if answer.upper() != "Y":
            print("Aborting the program.")
            exit()

    # create the base path and the sub path if they do not exist
    os.makedirs(os.path.dirname(full_path), exist_ok=True) # create the directory recursively and ignore the FileExistsError

    # save the data to the file using np.savetxt
    np.savetxt(full_path, data, header=header) # stack the arrays horizontally and write the header



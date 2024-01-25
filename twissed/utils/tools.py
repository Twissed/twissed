"""tools.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""

import os
import warnings
import functools
from typing import Optional


def make_executable(path: str) -> None:
    """Make file executable.

    Args:
        path (str): Path of the file to make executable.
    """
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2  # copy R bits to X
    os.chmod(path, mode)


def create_folder(folder_path: str, verbose: Optional[bool] = True) -> None:
    """Create a folder at the given path

    Args:
        folder_path (str): folder path+name
    """
    os.mkdir(folder_path)
    if verbose:
        print(f"Folder '{folder_path}' created")


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        warnings.warn(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter("default", DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func

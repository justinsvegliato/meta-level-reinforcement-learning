from datetime import datetime
import subprocess
from typing import Optional
import numpy as np

import silence_tensorflow.auto  # noqa
import tensorflow as tf


def one_hot(x: int, n: int) -> np.array:
    """
    One-hot encode a vector of integers.

    Args:
        x: The value to encode.
        n: The number of possible values.
    """
    vec = np.zeros(n)
    vec[x] = 1
    return vec


def time_id() -> str:
    """ Returns an id string based on the current time. """
    now = datetime.now()

    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    min = now.minute
    second = now.second

    return f'{second:02d}-{min:02d}-{hour:02d}-{day:02d}-{month:02d}-{year:02d}'


def get_current_git_commit(directory: str = '.') -> Optional[str]:
    """
    Get the sha of the current git commit from the given directory.
    If the directory is not a git repository or if there are no commits, return None.
    """
    try:
        # Check if the given directory is a git repository
        subprocess.check_output(["git", "-C", directory, "rev-parse", "--git-dir"],
                                stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        # The directory is not a git repository
        return None

    try:
        # Get the sha of the current commit
        sha = subprocess.check_output(["git", "-C", directory, "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL)
        return sha.decode('utf-8').strip()

    except subprocess.CalledProcessError:
        # There are no commits in the repository
        return None


def sanitize_dict(d: dict) -> dict:
    """Sanitizes a dictionary containing TensorFlow tensors and NumPy arrays.

    This function converts TensorFlow tensors and NumPy arrays to lists, so that
    the dictionary can be converted to a JSON object.

    Args:
        d: The dictionary to sanitize.

    Returns:
        A new dictionary with the same keys as the input dictionary, but with the
        values converted to lists.
    """
    sanitized_dict = {}
    for k, v in d.items():
        if isinstance(v, (np.ndarray, np.generic)):
            # Convert NumPy arrays to lists
            sanitized_dict[k] = np.squeeze(v).tolist()
        elif isinstance(v, (tf.Tensor, tf.Variable)):
            # Convert TensorFlow tensors to lists
            sanitized_dict[k] = np.squeeze(v.numpy()).tolist()
        elif isinstance(v, dict):
            # Recursively sanitize dictionaries
            sanitized_dict[k] = sanitize_dict(v)
        elif v is None:
            sanitized_dict[k] = "none"
        else:
            sanitized_dict[k] = v

        # If the value is a list with only one element, convert it to a scalar
        if isinstance(sanitized_dict[k], list):
            if len(sanitized_dict[k]) == 1:
                sanitized_dict[k] = sanitized_dict[k][0]

    return sanitized_dict

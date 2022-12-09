import time
import numpy as np


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


def time_id():
    """ Returns an id based on the current time. """
    return int(time.time() * 1e7)

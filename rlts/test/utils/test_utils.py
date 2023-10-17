from rlts.utils import get_current_git_commit, sanitize_dict

import subprocess
from unittest.mock import patch

import silence_tensorflow.auto  # noqa
import tensorflow as tf
import numpy as np


@patch('subprocess.check_output')
def test_not_git_repo(mock_check_output):
    mock_check_output.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd='git')
    assert get_current_git_commit() is None


@patch('subprocess.check_output')
def test_no_commits(mock_check_output):
    mock_check_output.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd='git')
    assert get_current_git_commit() is None


@patch('subprocess.check_output')
def test_has_commits(mock_check_output):
    mock_check_output.side_effect = [None, b'abc123\n']
    assert get_current_git_commit() == 'abc123'


def test_sanitize_dict():
    # Test with a dictionary containing a TensorFlow tensor
    d = {"tensor": tf.constant([1, 2, 3])}
    expected_output = {"tensor": [1, 2, 3]}
    assert sanitize_dict(d) == expected_output

    # Test with a dictionary containing a NumPy array
    d = {"array": np.array([4, 5, 6])}
    expected_output = {"array": [4, 5, 6]}
    assert sanitize_dict(d) == expected_output

    # Test with a dictionary containing a mixture of TensorFlow tensors and NumPy arrays
    d = {
        "tensor": tf.constant([1, 2, 3]),
        "array": np.array([4, 5, 6])
    }
    expected_output = {"tensor": [1, 2, 3], "array": [4, 5, 6]}
    assert sanitize_dict(d) == expected_output

    # Test with a dictionary containing non-tensor/array values
    d = {
        "string": "foo", "int": 123,
        "float": 3.14, "list": [1, 2, 3]
    }
    expected_output = {
        "string": "foo", "int": 123,
        "float": 3.14, "list": [1, 2, 3]
    }
    assert sanitize_dict(d) == expected_output

    # Test with an empty dictionary
    d = {}
    expected_output = {}
    assert sanitize_dict(d) == expected_output

    # Test with a np.array that has an unnecessary outer dimension
    d = {"array": np.array([[1, 2, 3]])}
    expected_output = {"array": [1, 2, 3]}
    assert sanitize_dict(d) == expected_output

    # Test with a tensor that has an unnecessary outer dimension
    d = {"tensor": tf.constant([[1, 2, 3]])}
    expected_output = {"tensor": [1, 2, 3]}
    assert sanitize_dict(d) == expected_output

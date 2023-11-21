from pathlib import Path
import os

import tensorflow as tf


def restrict_gpus(restricted_gpus: list):
    """
    Args:
        restricted: list of gpus not to use
    """

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in restricted_gpus])

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            print('Only using: ', ', '.join([f'GPU {i}' for i in restricted_gpus]))

            tf.config.set_visible_devices([
                gpu for i, gpu in enumerate(gpus)
                if i not in restricted_gpus
            ], 'GPU')

            # logical_gpus = tf.config.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print('Failed to set GPUS:', e)


def get_most_recently_modified_directory(path):
    """
    Returns the most recently modified directory in the specified folder using pathlib.
    
    Args:
        path (str): Path to the folder.
    
    Returns:
        Path: Path object to the most recently modified directory.
    """
    p = Path(path)
    
    # Get all directories in the path
    directories = [d for d in p.iterdir() if d.is_dir()]
    
    # Return None if there are no directories
    if not directories:
        return None
    
    # Sort the directories based on their modification time and get the most recent one
    most_recent_dir = max(directories, key=lambda d: d.stat().st_mtime)
    
    return most_recent_dir

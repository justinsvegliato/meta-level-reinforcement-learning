import tensorflow as tf


def restrict_gpus(restricted_gpus: list):
    """
    Args:
        restricted: list of gpus not to use
    """

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
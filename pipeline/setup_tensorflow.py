import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import tensorflow as tf
assert tf.__version__.startswith('2')

def dontUseWholeGPU():
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        print("No GPUs available.")
        return

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

dontUseWholeGPU()

import os
import random
import numpy as np
import tensorflow as tf

from scripts import utils
from scripts import settings

settings.seed_handler()

if __name__ == "__main__":
    print()
    (X, y) = utils.load_data(settings.Xs_dir, settings.ys_dir)[0]
    (X_test) = utils.load_test_data(settings.Xt_dir)[0]

    utils.mask_unused_gpus()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
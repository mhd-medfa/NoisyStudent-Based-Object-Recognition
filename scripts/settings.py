def init():
    global Xs_dir, ys_dir, Xt_dir
    Xs_dir = 'data/xs.npy'
    ys_dir = 'data/ys.npy'
    Xt_dir = 'data/xt.npy'

def seed_handler():
    seed_value=42

    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)

    import random
    random.seed(seed_value)

    import numpy as np
    np.random.seed(seed_value)

    import tensorflow as tf
    tf.random.set_seed(seed_value)

    tf.compat.v1.global_variables_initializer()
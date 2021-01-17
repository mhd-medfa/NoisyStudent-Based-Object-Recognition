import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

from scripts import settings
settings.seed_handler()
settings.init()

from scripts import utils
from scripts import model as m
from scripts.rand_augmentation import Rand_Augment

import os
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import tensorflow.keras as K
import datetime
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.layers import Conv2DTranspose, Reshape, Lambda, Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, Input, Activation, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def student(model, model_dir='student_model.h5'):

    # Pseudo-label unlabeled data in the teacher model
    x_train_student, y_train_student = utils.pseudo_labelling(model, X_data, y_data, X_test, threhold=0.9)

    """Prepare a student model
      Here, I will go with EffecientNetB2. As
      - model noise Dropout → Noise to the model

    Let the student model learn by giving noise with labeled + pseudo-label data """

    # Learning Student Model
    tf.compat.v1.global_variables_initializer()
    utils.mask_unused_gpus()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"

    # adding callbacks
    callback = []
    callback += [K.callbacks.LearningRateScheduler(m.decay, verbose=1)]
    callback += [K.callbacks.ModelCheckpoint(model_dir,
                                            save_best_only=True,
                                            mode='min'
                                            )]
    # tensorboard callback
    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # callback += [K.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]


    student_model, restored_flag = m.make_or_restore_model(name = 'Put your *.h5 file name', restore_flag=False, INPUT_SHAPE=(32, 32, 3))
    
    # restored_flag = True
    if restored_flag == False:
        # training model with mini batch using shuffle data
        batch_size=32
        steps_per_epoch = x_train_student.shape[0] // batch_size
        validation_steps = Xv_p.shape[0] // batch_size
        student_model.fit(utils.data_generator(x_train_student, y_train_student, batch_size, data_aug = True),
                        epochs=4,
                        steps_per_epoch = steps_per_epoch,
                        batch_size=batch_size,
                        validation_data = (Xv_p, Yv_p),
                        validation_steps = validation_steps,
                        callbacks=callback,
                        verbose=1
                        )

    utils.my_eval(student_model,Xv_p,Yv_p)
    # restored_flag = True
    if restored_flag == False:
        # training model with mini batch using shuffle data
        student_model.fit(x=X_p, y=Y_p,
        batch_size=8,
        validation_data=(Xv_p, Yv_p),
        epochs=1, shuffle=True,
        callbacks=callback,
        verbose=1
        )
    utils.my_eval(student_model,Xv_p,Yv_p)

    return student_model



if __name__ == "__main__":
    # loading data and using preprocess for training and validation dataset
    (X_data, y_data) = utils.load_data(settings.Xs_dir, settings.ys_dir)[0]
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

    Xt, X, Yt, Y = train_test_split(X_data, y_data, test_size=0.2)
 
    X_p, Y_p = m.preprocess_data(Xt, Yt)
    Xv_p, Yv_p = m.preprocess_data(X, Y)

    
    # adding callbacks
    callback = []
    callback += [K.callbacks.LearningRateScheduler(m.decay, verbose=1)]
    callback += [K.callbacks.ModelCheckpoint('model.h5',
                                            save_best_only=True,
                                            mode='min'
                                            )]
    # tensorboard callback
    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # callback += [K.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]


    model, restored_flag = m.make_or_restore_model(name = 'Put your *.h5 file name', restore_flag=False, INPUT_SHAPE=(32, 32, 3))
    
    # restored_flag = True
    if restored_flag == False:
        # training model with mini batch using shuffle data
        model.fit(x=X_p, y=Y_p,
        batch_size=8,
        validation_data=(Xv_p, Yv_p),
        epochs=2, shuffle=True,
        callbacks=callback,
        verbose=1
        )
    
    # restored_flag = True
    if restored_flag == False:
        # training model with mini batch using shuffle data
        model.fit(x=X_p, y=Y_p,
        batch_size=8,
        validation_data=(Xv_p, Yv_p),
        epochs=1, shuffle=True,
        callbacks=callback,
        verbose=1
        )
    
    student_model = student(model, 'student_model.h5')

    student2_model = student(student_model, 'student_model2.h5')


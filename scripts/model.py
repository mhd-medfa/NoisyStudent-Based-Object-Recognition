"""
This script has the method
preprocess_data(X, Y): and decay
and use transfer learning with VGG16 model
"""
import tensorflow.keras as K
import datetime
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.layers import Conv2DTranspose, Reshape, Lambda, Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, Input, Activation, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def preprocess_data(X, Y):
  """ This method has the preprocess to train a model """
  X_p = X
  # changind labels to one-hot representation
  Y_p = K.utils.to_categorical(Y, 9)
  return (X_p, Y_p)

def get_uncompiled_model(INPUT_SHAPE):
  # Getting the model without the last layers, trained with imagenet and with average pooling
  base_model = K.applications.EfficientNetB3(include_top=False,
                                                weights='imagenet', drop_connect_rate=0.4)
  # resize input
  IMG_SIZE = 300
  resize = K.Sequential([
    K.layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE)
  ])

  model= K.Sequential()
  # using upsamplign to get more data points and improve the predictions
  model.add(resize)

  model.add(base_model)
  model.add(K.layers.Flatten())
  model.add(K.layers.Dense(512, activation=('relu')))
  model.add(K.layers.Dropout(0.2))
  model.add(K.layers.Dense(256, activation=('relu')))
  model.add(K.layers.Dropout(0.2))
  model.add(K.layers.Dense(9, activation=('softmax')))
  return model

def decay(epoch):
  """ This method create the alpha"""
  # returning a very small constant learning rate
  return 0.001 / (1 + 1 * 30)

def get_compiled_model(INPUT_SHAPE=32):
  model = get_uncompiled_model(INPUT_SHAPE)
  model.compile(optimizer='adam', loss='categorical_crossentropy',
  metrics=['accuracy'])
  return model

def make_or_restore_model(name ='', restore_flag=True, INPUT_SHAPE=32):
  # Either restore the latest model, or create a fresh one
  # if there is no checkpoint available.
  if restore_flag==True:
    from google.colab import drive
    drive.mount('/content/gdrive')
    checkpoint_dir = "/content/gdrive/My Drive/innopolis-machine-learning-class/"
    checkpoints = [checkpoint_dir + "/" + name]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return K.models.load_model(latest_checkpoint), True # True if restored
  
  print("Creating a new model")
  return get_compiled_model(INPUT_SHAPE), False # False if created



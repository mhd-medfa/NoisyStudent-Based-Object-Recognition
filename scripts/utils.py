import os
import random
import numpy as np
import tensorflow as tf
import subprocess as sp
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from rand_augmentation import Rand_Augment
img_augment = Rand_Augment(Numbers=2, max_Magnitude=10)


"""## Loading the Data
Load images and labels.
"""
def load_data(xs='./xs.npy', ys='./ys.npy'):
    """
        Load the data:
            - 52,000 images to train the network and to evaluate 
            how accurately the network learned to classify images.
    """
    
    datasets = [xs, ys]
    output = []
    
    images = []
    labels = []

    print("Loading {}".format(datasets[0]))
    images = np.load(datasets[0])
    
    print("Loading {}".format(datasets[1]))
    labels = np.load(datasets[1])   
    
    output.append((images, labels))
    
    return output


def load_test_data(xt='./xt.npy'):
    """
        Load the data:
            - images with no labels.  
    """
    
    datasets = [xt]
    output = []  
    images = []

    print("Loading {}".format(datasets[0]))
    images = np.load(datasets[0])
      
    output.append((images,))
    
    return output


def mask_unused_gpus(leave_unmasked=1):
  ACCEPTABLE_AVAILABLE_MEMORY = 1024
  COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"

  try:
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    available_gpus = [i for i, x in enumerate(memory_free_values) if x > ACCEPTABLE_AVAILABLE_MEMORY]

    if len(available_gpus) < leave_unmasked: ValueError('Found only %d usable GPUs in the system' % len(available_gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, available_gpus[:leave_unmasked]))
  except Exception as e:
    print('"nvidia-smi" is probably not installed. GPUs are not masked', e)

def pseudo_labelling(model, xs, ys, xt, threhold=0.9):
  """
  Pseudo-label unlabeled data in the teacher model
      First, prepare an image for attaching a pseudo label. As a detailed procedure Make unlabeled images
      into numpy arrays Add a pseudo label to an unlabeled image Leave only pseudo-label data above a certain 
      threshold Align the number of data for each label It will be. 
  """
  x_train_9,x_test_9, y_train_9,y_test_9 = train_test_split(xs, ys, test_size=0.2)

  y_train_9 = to_categorical(y_train_9)
  y_test_9 = to_categorical(y_test_9)

  # ============Add a pseudo label to an unlabeled image============

  x_train_imgnet = xt[:-1]
  #Batch size setting
  batch_size = 10
  #How many steps
  step = int(x_train_imgnet.shape[0] / batch_size)
  print(step)

  #Empty list for pseudo labels
  y_train_imgnet_dummy = []

  for i in range(step):
      # Extract image data for batch size
      x_temp = x_train_imgnet[batch_size*i:batch_size*(i+1)]
      #normalization
      x_temp = x_temp
      #inference
      temp = model.predict(x_temp)
      #Add to empty list
      y_train_imgnet_dummy.extend(temp)

  #List to numpy array
  y_train_imgnet_dummy = np.array(y_train_imgnet_dummy)

  # ============Leave only pseudo-label data above a certain threshold============
  #Thresholding
  y_train_imgnet_dummy_th =  y_train_imgnet_dummy[np.max(y_train_imgnet_dummy, axis=1) > threhold]
  x_train_imgnet_th = x_train_imgnet[np.max(y_train_imgnet_dummy, axis=1) > threhold]

  #from onehot vector to class index
  y_student_all_dummy_label = np.argmax(y_train_imgnet_dummy_th, axis=1)

  #Count the number of each class of pseudo-labels
  u, counts = np.unique(y_student_all_dummy_label, return_counts=True)
  print(u, counts)

  #Calculate the maximum number of counts
  student_label_max =  max(counts)

  #Separate numpy array for each label
  y_student_per_label = []
  y_student_per_img_path = []

  for i in range(9):
      temp_l = y_train_imgnet_dummy_th[y_student_all_dummy_label == i]
      print(i, ":", temp_l.shape)
      y_student_per_label.append(temp_l)
      temp_i = x_train_imgnet_th[y_student_all_dummy_label == i]
      print(i, ":", temp_i.shape)
      y_student_per_img_path.append(temp_i)

  #Copy data for maximum count on each label
  y_student_per_label_add = []
  y_student_per_img_add = []

  for i in range(9):
      num = y_student_per_label[i].shape[0]
      temp_l = y_student_per_label[i]
      temp_i = y_student_per_img_path[i]
      add_num = student_label_max - num
      q, mod = divmod(add_num, num)
      print(q, mod)
      temp_l_tile = np.tile(temp_l, (q+1, 1))
      temp_i_tile = np.tile(temp_i, (q+1, 1, 1, 1))
      temp_l_add = temp_l[:mod]
      temp_i_add = temp_i[:mod]
      y_student_per_label_add.append(np.concatenate([temp_l_tile, temp_l_add], axis=0))
      y_student_per_img_add.append(np.concatenate([temp_i_tile, temp_i_add], axis=0))

  #Check the count number of each label
  print([len(i) for i in y_student_per_label_add])

  #Merge data for each label
  student_train_img = np.concatenate(y_student_per_img_add, axis=0)
  student_train_label = np.concatenate(y_student_per_label_add, axis=0)

  # Combined with the original training data numpy array
  x_train_student = np.concatenate([x_train_9, student_train_img], axis=0)
  y_train_student = np.concatenate([y_train_9, student_train_label], axis=0)

  return x_train_student, y_train_student

def my_eval(model,x,t):
    """
      model: Model to be evaluated,
      x: Image to be predicted
      shape = (batch, 32,32,3)
      t: label of one-hot representation"""
    ev = model.evaluate(x,t)
    print("loss:" ,end = " ")
    print(ev[0])
    print("acc: ", end = "")
    print(ev[1])

# Data generator definition
def get_random_data(x_train_i, y_train_i, data_aug):
  x = tf.keras.preprocessing.image.array_to_img(x_train_i)

  if data_aug:
      seed_image = img_augment(x)
      seed_image = tf.keras.preprocessing.image.img_to_array(seed_image)

  else:
      seed_image = x_train_i

  seed_image = seed_image

  return seed_image, y_train_i

def data_generator(x_train, y_train, batch_size, data_aug):
  '''data generator for fit_generator'''
  n = len(x_train)
  i = 0
  while True:
      image_data = []
      label_data = []
      for b in range(batch_size):
          if i==0:
              p = np.random.permutation(len(x_train))
              x_train = x_train[p]
              y_train = y_train[p]
          image, label = get_random_data(x_train[i], y_train[i], data_aug)
          image_data.append(image)
          label_data.append(label)
          i = (i+1) % n
      image_data = np.array(image_data)
      label_data = np.array(label_data)
      yield image_data, label_data
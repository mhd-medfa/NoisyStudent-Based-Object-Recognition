import os
import random
import numpy as np
import tensorflow as tf
import subprocess as sp
import os
import settings


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

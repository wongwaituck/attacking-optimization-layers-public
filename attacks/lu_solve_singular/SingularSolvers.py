import torch
import scipy.linalg
import torch.nn as nn
import torch.nn.functional as fnn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import random
import sys
import math
import numpy as np


tf.random.set_seed(0)
torch.manual_seed(0)

for i in range(100):
    # initialize system of equations
    a1 = [9, 1, -9]
    x = np.random.randint(-10, 10)
    y = np.random.randint(-10, 10)
    z = np.random.randint(-10, 10)
    mult = 2
    A = torch.tensor([a1, [x, y, z], [mult*x + 18, mult*y + 2, mult*z - 18]]).double()
    print(A)
    b = torch.tensor([[1], [1], [1]]).double()

    try:
        torchLU = torch.lu(A)
        print(torch.lu_solve(b, *torchLU))
    except Exception as e:
        print("Torch linalg failed:")
        print(e)
    try:
        tfLU = tf.linalg.lu(A)
        print("TENSORFLOW SOLUTION!!!!!!")
        print(tf.linalg.lu_solve(*tfLU, b))
    except Exception as e:
        print("TensorFlow linalg failed:")
        print(e)
    

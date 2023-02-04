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


# create singular A
A = torch.tensor([
        [9,   1, -9],
        [-13, 8,  33],
        [29,  7, -21]]).double()

# rhs
b = torch.tensor([[3],[1],[1]]).double()

a, s, d = torch.linalg.svd(A)
print("Determinant: ")
print(A.det().item())
print(s)
print(A)

# print(torch.linalg.solve(A, b))

# pytorch behavior
try:
    torchLU = torch.lu(A)
    print(torch.lu_solve(b, *torchLU))
except Exception as e:
    print("Torch linalg failed:")
    print(e)

# tensorflow behavior
try:
    tfLU = tf.linalg.lu(A)
    print(tf.linalg.lu_solve(*tfLU, b))
except Exception as e:
    print("TensorFlow linalg failed:")
    print(e)
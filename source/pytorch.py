#!/usr/bin/env python
# coding: utf-8

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

# Check PyTorch version
print(torch.__version__)

# Initializing tensors
one_d_tensor = torch.tensor([1])
print(one_d_tensor)
print(one_d_tensor.shape)

two_d_tensor = torch.tensor([[1, 2], [3, 4]])
print(two_d_tensor)
print(two_d_tensor.shape)

three_d_tensor = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                               [[10, 11, 12], [13, 14, 15], [16, 17, 18]]])
print(three_d_tensor)
print(three_d_tensor.shape)

# Randomly initializing tensors
torch.manual_seed(42)
random_1d_tensor = torch.randn(3)
print(random_1d_tensor)

random_2d_tensor = torch.randn(3,3)
print(random_2d_tensor)

random_3d_tensor = torch.randn(2,3,3)
print(random_3d_tensor)

# Mathematical Operations
a = torch.tensor(2)
b = torch.tensor(1)
print(a, b)
print(a+b)
print(b-a)
print(a*b)
print(a/b)

# Matrix Operations
torch.manual_seed(42)
a = torch.randn(3,3)
b = torch.randn(3,3)
print(a)
print(b)

# Matrix addition
addition = torch.add(a,b)
print(addition)

# Matrix subtraction
subtraction = torch.sub(a,b)
print(subtraction)

# Matrix multiplication
dot_product = torch.mm(a,b)
print(dot_product)

# Elementwise multiplication
elementwise_multiplication = torch.mul(a,b)
print(elementwise_multiplication)

# Matrix division
division = torch.div(a,b)
print(division)

# Transpose of a matrix
print(a, '\n')
print(torch.t(a))

# Reshaping tensors
a = torch.randn(2,4)
print(a)
print(a.shape)

b = a.reshape(1,8)
print(b)
print(b.shape)

# Sum and average of tensor values
print(a.sum())
print(torch.mean(a))

# Convert NumPy arrays to PyTorch tensors
np_array = np.array([[1,2],[3,4]])
print(np_array)
tensor_from_array = torch.from_numpy(np_array)
print(tensor_from_array)

# Convert PyTorch tensors to NumPy arrays
tensor = torch.tensor([[1,2],[3,4]])
print(tensor)
array_from_tensor = tensor.numpy()
print(array_from_tensor)

# Check GPU availability
if torch.cuda.is_available():
    print('GPU is available')
else:
    print('GPU is not available')

# Operations with tensors on CPU
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[1, 2], [3, 4]])
c = a + b
print(c)

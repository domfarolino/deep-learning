# DL6B.py CS5173/6073 cheng 2019
# from geron's hands-on ML chapter 9
# tensorflow on linear regression with normal equation
# with the housing data
# many warnings on usage 
# Usage: python DL6B.py

import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

graph = tf.get_default_graph()
operations = graph.get_operations()
print(operations)
print(len(operations))
print(operations[9])

with tf.Session() as sess:
    theta_value = theta.eval()
    print(theta_value)

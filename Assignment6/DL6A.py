# DL6A.py CS5173/6073 cheng 2019
# from geron's hands-on ML chapter 9
# four ways to use tensorflow sessions
# prints operations and the details of one of them
# warning messages may show up for depreceted usages
# Usage: python DL6A.py

import numpy as np
import tensorflow as tf

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

graph = tf.get_default_graph()
operations = graph.get_operations()
print(operations)
print(len(operations))
print(operations[9])

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
    print(result)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    result = f.eval()
    print(result)

sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
sess.close()


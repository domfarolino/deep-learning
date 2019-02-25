# DL6F.py CS5173/6073 cheng 2019
# tensorflow GradientDescentOptimizer on softmax multiclass linear model
# with the mnist data
# many warnings on usage 
# Usage: python DL6F.py

import tensorflow as tf

######
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
######

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batchSz=100
W = tf.Variable(tf.random_normal([784, 10],stddev=.1))
b = tf.Variable(tf.random_normal([10],stddev=.1))

img = tf.placeholder(tf.float32, [batchSz,784])
ans = tf.placeholder(tf.float32, [batchSz, 10])

prbs = tf.nn.softmax(tf.matmul(img, W) + b)
xEnt = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(prbs),
reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.5).minimize(xEnt)
numCorrect= tf.equal(tf.argmax(prbs,1), tf.argmax(ans,1))
accuracy = tf.reduce_mean(tf.cast(numCorrect, tf.float32))

graph = tf.get_default_graph()
operations = graph.get_operations()
print(operations)
print(len(operations))
print(operations[9])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#-------------------------------------------------
for i in range(1000):
  imgs, anss = mnist.train.next_batch(batchSz)
  sess.run(train, feed_dict={img: imgs, ans: anss})
sumAcc=0
for i in range(1000):
  imgs, anss= mnist.test.next_batch(batchSz)
  sumAcc+=sess.run(accuracy, feed_dict={img: imgs, ans: anss})
print(sumAcc/1000)

# DL9B.py CS5173/6073 cheng 2019
# restore graph and learned values from those saved by DL9A
# plot the 32 conv1 level features

import tensorflow as tf

saver = tf.train.import_meta_graph("./my_fashion_mnist_model.meta")  #// the computation graph
graph = tf.get_default_graph()
operations = graph.get_operations();
print([v.name for v in tf.trainable_variables()])  #   // those parameters learned
ker1 = graph.get_tensor_by_name("conv1/kernel:0")  #   // first convolutional layer
ker2 = graph.get_tensor_by_name("conv2/kernel:0")  #   // second convolutional layer
sess = tf.Session()
saver.restore(sess, tf.train.latest_checkpoint("./"))##    // the learned parameters
features1 = sess.run(ker1)  #// get weights of the first convolutional layer
features2 = sess.run(ker2)  
print(features2.shape) # // how many values are in the second convolutional layer!

import numpy as np
import matplotlib.pyplot as plt

def plot_digits(instances, images_per_row=4, **options): # // borrowed from DL1B.py
 #// instances should be an array of 32 arrays of size 9 each for the 32 conv1 features
    size = 3
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = 'bone')
    plt.axis("off")

plt.figure(figsize=(7, 4))
plt.subplot(121)
plot_digits(features1.flatten().reshape(9,32).transpose())  # // you fill the new shape
plt.title("conv1", fontsize=16)

plt.show()


# DL1B.py CS5173/6073 2019 cheng
# show "eigenfaces" for the digits

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1)
    mnist.target = mnist.target.astype(np.int64)
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')


from sklearn.model_selection import train_test_split

X = mnist["data"]
y = mnist["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.decomposition import PCA

pca = PCA(n_components = 154)
X_reduced = pca.fit_transform(X_train)

def plot_digits(instances, images_per_row=5, **options):
    size = 28
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
plot_digits(pca.components_[:25])
plt.title("Features", fontsize=16)

plt.show()


# DL5B2.py CS5173/6073 cheng 2019
# from agenon and uses sklearn's TSNE on MNIST
# plotting with digits
# execution time is also reported and it may take several minutes

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.uint8)

from sklearn.model_selection import train_test_split

X = mnist["data"]
y = mnist["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)


m = 10000
idx = np.random.permutation(60000)[:m]

X = mnist['data'][idx]
y = mnist['target'][idx]

selected = set()
while len(selected) < 3:
   selected.add(np.random.randint(10))
selected2 = list(selected)

idx2 = (y == selected2[0]) | (y == selected2[1]) | (y == selected2[2])
X_subset = X[idx2]
y_subset = y[idx2]

from sklearn.manifold import TSNE
import time

t0 = time.time()
tsne = TSNE(n_components=2)
X_subset_reduced = tsne.fit_transform(X_subset)
t1 = time.time()
print("t-SNE took {:.1f}s.".format(t1 - t0))

from sklearn.preprocessing import MinMaxScaler
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

def plot_digits(X, y, min_distance=0.05, figsize=(9, 9)):
    # Let's scale the input features so that they range from 0 to 1
    X_normalized = MinMaxScaler().fit_transform(X)
    # Now we create the list of coordinates of the digits plotted so far.
    # We pretend that one is already plotted far away at the start, to
    # avoid `if` statements in the loop below
    neighbors = np.array([[10., 10.]])
    # The rest should be self-explanatory
    plt.figure(figsize=figsize)
    cmap = mpl.cm.get_cmap("jet")
    digits = np.unique(y)
    for digit in digits:
        plt.scatter(X_normalized[y == digit, 0], X_normalized[y == digit, 1], c=[cmap(digit / 9)])
    plt.axis("off")
    ax = plt.gcf().gca()  # get current axes in current figure
    for index, image_coord in enumerate(X_normalized):
        closest_distance = np.linalg.norm(np.array(neighbors) - image_coord, axis=1).min()
        if closest_distance > min_distance:
            neighbors = np.r_[neighbors, [image_coord]]
            plt.text(image_coord[0], image_coord[1], str(int(y[index])),
               color=cmap(y[index] / 9), fontdict={"weight": "bold", "size": 16})

plot_digits(X_subset_reduced, y_subset)
plt.show()


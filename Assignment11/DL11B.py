# DL11B.py CS5173/6073 cheng 2019
# This program displays a randomly selected section of 100 words of vocabulary 
# generated from words.zip and ordered by frequency
# with their word2vec embeddings in a 128 dimensional space
# constructed with DL11A.py and saved in my_final_embeddings.npy
# using t-SNE with PCA initialization on a 2D plot
# You may fix the offset to 0 to have the most frequent 100 or 500 words plotted.
# Usage : python DL11B.py

import os
import zipfile

import numpy as np

WORDS_PATH = "datasets/words"

def fetch_words_data(words_path=WORDS_PATH):
    os.makedirs(words_path, exist_ok=True)
    zip_path = os.path.join(words_path, "words.zip")
    with zipfile.ZipFile(zip_path) as f:
        data = f.read(f.namelist()[0])
    return data.decode("ascii").split()

words = fetch_words_data()

from collections import Counter

vocabulary_size = 50000

vocabulary = [("UNK", None)] + Counter(words).most_common(vocabulary_size - 1)
vocabulary = np.array([word for word, _ in vocabulary])

final_embeddings = np.load("./my_final_embeddings.npy")

import matplotlib.pyplot as plt

def plot_with_labels(low_dim_embs, labels):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

from sklearn.manifold import TSNE

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 100
offset = np.random.randint(2000)
low_dim_embs = tsne.fit_transform(final_embeddings[offset:plot_only+offset:,:])
labels = [vocabulary[i+offset] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)

plt.show()

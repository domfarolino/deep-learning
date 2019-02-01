# DL2A.py CS5173/6073 cheng 2019
# from agenon and uses sklearn's LogisticRegression
# This program reads the Iris data and prints out part of the data
# only petal width is used for Iris Versicolor classification

from sklearn import datasets
iris = datasets.load_iris()
print(iris.keys())
print(iris.DESCR)
import numpy as np

X = iris["data"][:, 2:-1]  # petal length
y = (iris["target"] == 1).astype(np.int)  # 1 if Iris-Versicolor, else 0

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver="liblinear")
log_reg.fit(X, y)
print(log_reg.coef_)

X_new = np.linspace(0, 10, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)

import matplotlib.pyplot as plt

plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Versicolor")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris-Versicolor")
plt.show()

X_new = np.linspace(0, 10, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= .5][0]

plt.figure(figsize=(8, 5))
plt.plot(X[y==0], y[y==0], "bs")
plt.plot(X[y==1], y[y==1], "g^")
plt.plot([decision_boundary, decision_boundary], [-1, 10], "k:", linewidth=2)
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Versicolor")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris-Versicolor")
plt.text(decision_boundary+0.02, 0.15, "Decision  boundary", fontsize=14, color="k", ha="center")
plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
plt.xlabel("Petal length (cm)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 8, -0.02, 1.02])
plt.show()


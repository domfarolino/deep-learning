# DL2B.py CS5173/6073 cheng 2019
# from agenon and uses sklearn's LogisticRegression
# petal length and petal width are used for Iris Virginica classification

from sklearn import datasets
iris = datasets.load_iris()
import numpy as np

X = iris["data"][:, (0, 1)]  # sepal length, sepal width
y = (iris["target"] == 1).astype(np.int)  # 1 if Iris-Versicolor, else 0

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver="liblinear", C=10**10)
log_reg.fit(X, y)
print(log_reg.coef_)

x0, x1 = np.meshgrid(
        np.linspace(2, 11, 500).reshape(-1, 1),
        np.linspace(1, 5, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = log_reg.predict_proba(X_new)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(X[y==0, 0], X[y==0, 1], "bs")
plt.plot(X[y==1, 0], X[y==1, 1], "g^")

zz = y_proba[:, 1].reshape(x0.shape)
contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)

left_right = np.array([2.9, 7])
boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]

plt.clabel(contour, inline=1, fontsize=12)
plt.plot(left_right, boundary, "k--", linewidth=3)
plt.text(3.5, 1.5, "Not Iris-Versicolor", fontsize=14, color="b", ha="center")
plt.text(6.5, 2.3, "Iris-Versicolor", fontsize=14, color="g", ha="center")
plt.xlabel("Sepal length", fontsize=14)
plt.ylabel("Sepal width", fontsize=14)
plt.axis([2, 11, 1, 5])
plt.show()

import numpy as np
from sklearn.svm import SVC
import mglearn
import matplotlib.pyplot as plt
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D,axes3d

data = datasets.make_moons(n_samples=2000, noise=0.1)
X, y = data

points = np.array([])
for i in range(-3,4):
    points = np.append(points, np.linspace(10**i, 10**(i+1), num=5, endpoint=False)) #doesn't need to ravel()

count = 0
accuracy_best = 0

for C in points:
    for gamma in points:
        clf = SVC(gamma=gamma, C=C)
        clf.fit(X,y)
        # accuracy = clf.score(test_data,test_label)
        count += 1
        if C > 0.00000023:
            C0 = C
            gamma0 = gamma
            accuracy_best = accuracy
        else:
            continue
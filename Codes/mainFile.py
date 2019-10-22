from sklearn.svm import SVC
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from bonnerLib import dfContour, df3D
# %matplotlib.inline

data_path = r'C:\Data\CS6362\Assignment2'

if __name__ != "__main__":
    """%%% Load the data %%%"""
    data = datasets.make_moons(n_samples=2000, noise=0.1)
    X,y = data
    plt.figure()
    plt.suptitle('Moons data sample')
    colors = np.array(['r','b'])
    # plt.scatter(X[:,0],X[:,1],color = colors[y],s=3)
    # plt.show()

    """%%% Set Params, fit and RBF kernel %%%"""
    clf = SVC(gamma=1.0, C=1.0)
    clf.fit(X,y)

    """%%% Plot figure 1 and 2"""
    dfContour(clf, data)
    df3D(clf, data)
    plt.show()

    """%%% Make predictions"""
    # clf.predict(Xtest)
    # clf.decision_function(Xtest)

if __name__ == "__main__":
    print('Taking train and test data...')
    train_set = datasets.make_moons(n_samples=200, noise=0.4,random_state=1)
    train_data, train_label = train_set
    test_set = datasets.make_moons(n_samples=2000, noise=0.4,random_state=0)
    test_data, test_label = test_set
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # colors = np.array(['r','b'])
    # plt.scatter(train_data[:, 0], train_data[:, 1], color=colors[train_label], s=3)
    # plt.title("train data",fontsize=15)
    # plt.subplot(1, 2, 2)
    # colors = np.array(['r', 'b'])
    # plt.scatter(test_data[:, 0], test_data[:, 1], color=colors[test_label], s=3)
    # plt.title("test data",fontsize=15)
    # plt.show()

if __name__ != "__main__":
    print("save...")
    save_path = os.path.join(data_path,'test_set.bin')
    # print(save_path)
    # with open(save_path,'wb') as pfile:
    #     pickle.dump(test_set, pfile)

if __name__ != "__main__":
    print("loading the same set of data...")
    save_path = os.path.join(data_path, 'test_set.bin')
    with open(save_path,'rb') as pfile:
        test_set = pickle.load(pfile)

    save_path = os.path.join(data_path, 'train_set.bin')
    with open(save_path,'rb') as pfile:
        train_set = pickle.load(pfile)

    train_data, train_label = train_set
    test_data, test_label = test_set
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # colors = np.array(['r','b'])
    # plt.scatter(train_data[:, 0], train_data[:, 1], color=colors[train_label], s=3)
    # plt.title("train data")
    # plt.subplot(1, 2, 2)
    # colors = np.array(['r', 'b'])
    # plt.scatter(test_data[:, 0], test_data[:, 1], color=colors[test_label], s=3)
    # plt.title("test data")
    # plt.show()

# C = np.logspace(4.0, 4.01, num=5, endpoint=True, base=10)
# # plt.plot(np.log10(C))
# # plt.show()
# print(np.log10(C))
# print(C)
"""%%%% Greedy search %%%%"""
if __name__ == "__main__":
    # start = -4.0
    # end = -
    # n = 0
    # points = []
    #
    # while n < 10:
    #     # C = np.logspace(start, end, num=5, endpoint=False, base=10)
    #     # points.append(C)
    #     print(start, end)
    #     start = end
    #     end += 1
    #     n += 1
    #
    #
    # print(np.array(points).shape)
    # points = np.array(points).ravel()

    points = np.array([])

    for i in range(-3,4):
        points = np.append(points, np.linspace(10**i, 10**(i+1), num=5, endpoint=True)) #doesn't need to ravel()

    count = 0
    accuracy_best = 0

    for c in points:
        for gamma in points:
            clf = SVC(gamma=gamma, C=c)
            clf.fit(train_data,train_label)
            accuracy = clf.score(test_data,test_label)
            count += 1
            if accuracy>accuracy_best:
                C0 = c
                gamma0 = gamma
                accuracy_best = accuracy
            else:
                continue

    print("best fit:", C0, gamma0, count, accuracy_best) #best fit: 1.0 1.0 1225 0.867
    #1000.0 0.09942600739529568 2500 0.87
    # plt.plot(np.log10(points))
    # plt.show()

if __name__ != "__main__":
    C0 = 1.0
    gamma0 = 1.0
    accuracy_best = 0.867

"""Plot best model"""
if __name__ != "__main__":
    clf = SVC(gamma=gamma0, C=C0)
    clf.fit(train_data,train_label)
    dfContour(clf, test_set)
    plt.title("Decision boundary with lowest test error: {}%".format((1-accuracy_best)*100))
    df3D(clf, test_set)
    # clf.decision_function(Xtest)
    plt.show()

"""Error plot for fixed γ"""
if __name__ == "__main__":
    test_err = []
    train_err = []
    start = np.log10(C0) - 3 #todo find values here
    end = np.log10(C0) + 3 #todo
    # following C are both the same
    C = np.logspace(start, end, num=100, endpoint=True, base=10)
    # C = np.power(10, np.linspace(start, end, num=100))
    plt.subplot(2,1,1)
    plt.plot(C,label= "C")
    plt.legend(loc='upper left', fontsize=15)
    plt.subplot(2,1,2)
    plt.plot(np.log10(C),label = "log10(C)")
    plt.legend(loc='upper left', fontsize=15)
    # plt.plot(C)
    plt.suptitle("C evenly spaced in log scale", fontsize=15)
    plt.show()


if __name__ == "__main__":
    for c in C:
        clf = SVC(gamma=gamma0,C=c)
        clf.fit(train_data,train_label)
        err = 1 - clf.score(train_data,train_label)
        train_err.append(err)
        err = 1 - clf.score(test_data, test_label)
        test_err.append(err)

    plt.plot(np.log10(C),train_err,'b',linewidth=2,label="Train error")
    plt.plot(np.log10(C),test_err,'g',linewidth=2,label='Test error')
    plt.axvline(np.log10(C0))
    plt.xlabel("log10(C)",fontsize=15)
    plt.ylabel("Error(a.u.)",fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.title("Train and test error for γ: {:.4f}".format(gamma0))
    plt.show()

if __name__ == "__main__":
    start = np.log10(C0) - 3
    end = np.log10(C0) + 3
    C = np.logspace(start,end,num=7,endpoint=True,base=10)
    # C = np.power(10, np.linspace(start, end, num=7))

    for i,c in enumerate(C):
        clf = SVC(gamma=gamma0,C=c)
        clf.fit(train_data,train_label)
        plt.subplot(4,2,i+1)
        dfContour(clf, train_set)
        plt.title("C = {}".format(c))
    plt.suptitle("Decision function with varied C")
    plt.tight_layout(pad=0.1, w_pad=0.5, h_pad=0.7)
    plt.show()

"""Error plot for fixed C"""
if __name__ == "__main__":
    test_err = []
    train_err = []
    start = np.log10(gamma0) - 3
    end = np.log10(gamma0) + 3
    gamma = np.logspace(start, end, num=100, endpoint=True, base=10)
    for g in gamma:
        clf = SVC(gamma=g,C=C0)
        clf.fit(train_data,train_label)
        err = 1 - clf.score(train_data,train_label)
        train_err.append(err)
        err = 1 - clf.score(test_data, test_label)
        test_err.append(err)
    plt.plot(np.log10(gamma),train_err,'b',linewidth=2,label="Train error")
    plt.plot(np.log10(gamma),test_err,'g',linewidth=2,label='Test error')
    plt.axvline(np.log10(gamma0))
    plt.xlabel("log10(γ)",fontsize=15)
    plt.ylabel("Error(a.u.)",fontsize=15)
    plt.legend(loc='upper left', fontsize=15)
    plt.title("Train and test error for C: {:.2f}".format(C0))
    plt.show()

if __name__ == "__main__":
    start = np.log10(gamma0) - 3
    end = np.log10(gamma0) + 3
    gamma = np.logspace(start,end,num=7,endpoint=True,base=10)

    for i,g in enumerate(gamma):
        clf = SVC(gamma=g,C=C0)
        clf.fit(train_data,train_label)
        plt.subplot(4,2,i+1)
        dfContour(clf, train_set)
        plt.title("γ = {:.4f}".format(g))
    plt.suptitle("Decision function with varied γ")
    plt.tight_layout(pad=0.1, w_pad=0.5, h_pad=0.7)
    plt.show()
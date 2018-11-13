import mnist
import numpy as np
import collections
import math

def processamento():
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()

    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    train_images = train_images.reshape((train_images.shape[0], train_images.shape[1] * train_images.shape[2]))
    test_images = test_images.reshape((test_images.shape[0], test_images.shape[1]*test_images.shape[2]))

    return train_images, train_labels, test_images, test_labels

def pi(Y):
    Pi = collections.Counter(Y)

    for i in range(len(Pi)):
        Pi[i] = Pi[i]/len(Y)

    Pi = collections.OrderedDict(sorted(Pi.items()))

    return Pi

def mis(X, Y):
    mi = []
    y = list(set(Y))
    for i in range(len(set(Y))):
        aux = X[np.where(Y == i)]
        mi.append(np.mean(aux, axis=0))

    return np.array(mi)

def sigma(X):
    sigma_j = np.cov(X.T)
    sigma_j = sigma_j + (np.identity(sigma_j.shape[0]) * 0.00001)
    return sigma_j

def funBeta(sigma_j, mi, Pi):
    pz = list(Pi.values())

    gama = [-mi[i].T.dot(np.linalg.inv(sigma_j).dot(mi[i])) + np.log(pz[i]) for i in range(len(mi))]

    beta = [np.linalg.inv(sigma_j).dot(mi[i]) for i in range(len(mi))]

    beta = np.array(beta)

    gama = np.array(gama)

    beta = np.c_[gama, beta]

    return beta

def softmax(X, beta):
    p_x = []
    for i in range(beta.shape[0]):
        p_x.append(math.exp(X.dot(beta[i])))
    aux = sum(p_x)
    p_x = np.array(p_x)
    p_x = p_x/aux
    predicao = list(p_x).index(max(p_x))
    return predicao


def main():
    X, Y, X_test, Y_test = processamento()
    Pi = pi(Y)
    mi = mis(X, Y)
    sigma_j = sigma(X)
    beta = funBeta(sigma_j, mi, Pi)
    X_test = np.c_[np.ones(X_test.shape[0]), X_test]

    predicao = []
    acc = 0
    for i in range(X_test.shape[0]):
        predicao.append(softmax(X_test[i], beta))
        if predicao[i] == Y_test[i]:
            acc +=1

    acc = (acc/Y_test.shape[0])*100
    print(acc)

if __name__ == '__main__':
    main()
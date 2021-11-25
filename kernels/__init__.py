import scipy
import scipy.linalg
import torch
import torchvision
import torch.nn.functional as F
import torchvision.datasets as dst
import numpy as np
from tqdm.auto import tqdm, trange
import pylab as plt
import numpy.linalg as la
import pandas as pd
from enum import Enum


def load_fmnist():
    train_ds = dst.FashionMNIST(root='~/tmp/data', train=True,
                                download=True, transform=None)
    test_ds = dst.FashionMNIST(root='~/tmp/data', train=False,
                               download=True, transform=None)

    def to_xy(dataset):
        X = np.array(dataset.data) / 255.0
        Y = np.array(dataset.targets)
        return X, Y

    X_tr, Y_tr = to_xy(train_ds)
    X_te, Y_te = to_xy(test_ds)
    return X_tr, Y_tr, X_te, Y_te


def load_fmnist_all():
    X_tr, Y_tr, X_te, Y_te = load_fmnist()
    X = np.concatenate((X_tr, X_te)).astype(np.float32)
    Y = np.concatenate((Y_tr, Y_te))
    return X, Y


def load_mnist():
    train_ds = dst.MNIST(root='~/tmp/data', train=True,
                                download=True, transform=None)
    test_ds = dst.MNIST(root='~/tmp/data', train=False,
                               download=True, transform=None)

    def to_xy(dataset):
        X = np.array(dataset.data) / 255.0
        Y = np.array(dataset.targets)
        return X, Y

    X_tr, Y_tr = to_xy(train_ds)
    X_te, Y_te = to_xy(test_ds)
    return X_tr, Y_tr, X_te, Y_te

def load_mnist_all():
    X_tr, Y_tr, X_te, Y_te = load_mnist()
    X = np.concatenate((X_tr, X_te)).astype(np.float32)
    Y = np.concatenate((Y_tr, Y_te))
    return X, Y

def load_cifar_np():
    train_ds = dst.CIFAR10(root='~/tmp/data', train=True,
                           download=True, transform=None)
    test_ds = dst.CIFAR10(root='~/tmp/data', train=False,
                          download=True, transform=None)

    def to_xy(dataset):
        X = np.array(dataset.data) / 255.0
        Y = np.array(dataset.targets)
        return X, Y

    X_tr, Y_tr = to_xy(train_ds)
    X_te, Y_te = to_xy(test_ds)
    return X_tr, Y_tr, X_te, Y_te

def load_cifar_all():
    X_tr, Y_tr, X_te, Y_te = load_cifar_np()
    X = np.concatenate((X_tr, X_te)).astype(np.float32)
    Y = np.concatenate((Y_tr, Y_te))
    return X, Y

def distXX(X):
    ''' Returns d(X, X), pairwise distances between all points.
        d[i, j] = ||X_i - X_j||^2
    '''

    def flatten(x):
        return x.reshape((x.shape[0], -1))

    X = flatten(X)

    ## thanks to vaishaal
    D = X.dot(X.T)
    D *= -2
    D += np.linalg.norm(X, axis=1)[:, np.newaxis] ** 2
    D += np.linalg.norm(X, axis=1)[np.newaxis, :] ** 2
    return D


class KernelType(Enum):
    GAUSSIAN = 1
    LAPLACE = 2

class KernelPredictor():
    def __init__(self, X, Y, num_classes=10):
        self.D = distXX(X)
        self.y = Y
        self.y_tr = Y
        self.Yenc = F.one_hot(torch.Tensor(Y).long(), num_classes).float().numpy()
        self.dim = np.prod(X.shape[1:])  # 28 x 28
        self.nc = num_classes

    def set_test(self, I_te):
        '''
        Sets the indices of the test set.
        '''
        self.I_te = I_te

    def d_to_K(self, D, ktype=KernelType.GAUSSIAN, sigma=0.1):
        '''
            distances --> kernel
        '''
        s = sigma * np.sqrt(self.dim)
        if ktype == KernelType.GAUSSIAN:
            return np.exp(-0.5 * D / (s ** 2))
        else:
            return np.exp(-np.sqrt(D.clip(0)) / s)

    def train(self, I_tr, y_tr=None, ktype=KernelType.GAUSSIAN, sigma=0.1):
        if y_tr is None:
            y_tr = self.y[I_tr]
        Y = np.eye(self.nc)[y_tr]
        Ktr = self.d_to_K(self.D[I_tr, :][:, I_tr], ktype, sigma)
        model = scipy.linalg.solve(Ktr, Y, sym_pos=True, check_finite=False)
        return model

    def predict(self, I_tr, I_te, ktype = KernelType.GAUSSIAN, sigma=0.1):
        model = self.train(I_tr, ktype=ktype, sigma=sigma)
        Kpred = self.d_to_K(self.D[I_te, :][:, I_tr], ktype, sigma)
        yhats = Kpred.dot(model)
        preds = np.argmax(yhats, axis=1)
        return preds

    def predict_Kte(self, I_tr, Kte, ktype=KernelType.GAUSSIAN, sigma=0.1):
        model = self.train(I_tr, ktype, sigma)
        yhats = Kte.dot(model)
        preds = np.argmax(yhats, axis=1)
        return preds

    def train_pred(self, I_tr, y_tr=None, I_te=None, ktype = KernelType.GAUSSIAN, sigma=0.1):
        if y_tr is None:
            y_tr = self.y[I_tr]
        if I_te is None:
            I_te = self.I_te

        model = self.train(I_tr, y_tr=y_tr, ktype=ktype, sigma=sigma)
        Kte = self.d_to_K(self.D[I_te, :][:, I_tr], ktype, sigma)
        yhats = Kte.dot(model)
        preds = np.argmax(yhats, axis=1)
        return preds

####

class CifarKernel():
    def __init__(self):
        pass

    def load_from_npz(self, cifar_kernel):
        self.K_tr = cifar_kernel["K_train"]
        self.K_te = cifar_kernel["K_test"]
        self.y_tr = cifar_kernel["y_train"]
        self.y_te = cifar_kernel["y_test"]

    def load(self):
        fname = '/home/jupyter/hardness/s3_cache/cifar_kernels/cifar10_myrtle10_50k.npz'
        cifar_kernel = np.load(fname)
        print(cifar_kernel.files)
        self.load_from_npz(cifar_kernel)

    def train(self, I_tr, y_tr):
        K = self.K_tr[I_tr, :][:, I_tr]
        y = np.eye(10)[y_tr]  # one hot
        model = scipy.linalg.solve(K, y, sym_pos=True, check_finite=False)
        return model

    def train_eval(self, I_tr, y_tr):
        model = self.train(I_tr, y_tr)  # n_tr x 10
        Kte = self.K_te[:, I_tr]
        yhat = Kte.dot(model)
        preds = np.argmax(yhat, axis=1)
        return model, yhat, preds

    def train_test(self, I_tr):
        return self.train_eval(I_tr, self.y_tr[I_tr])

    def train_pred(self, I_tr, y_tr=None):
        if y_tr is None:
            y_tr = self.y_tr[I_tr]

        model, yhat, preds = self.train_eval(I_tr, y_tr)
        return preds



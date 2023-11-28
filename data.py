import os
import json
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset
from torchvision.datasets import CIFAR10, MNIST
import torch.nn.functional as F

# DATASETS_FOLDER = os.environ["DATASETS"]
CIFAR_NUM_CLASSES = 10
CIFAR_NUM_CHANNELS = 3
CIFAR_IMG_SIZE = 32
CIFAR_NUM_PIXELS = CIFAR_NUM_CHANNELS * CIFAR_IMG_SIZE ** 2

DATASETS = ["cifar10",
            "cifar10-1k",
            "cifar10-2k",
            "cifar10-5k",
            "cifar10-10k",
            "cifar10-20k"]

def flatten(arr: np.ndarray):
    return arr.reshape(arr.shape[0], -1)


def unflatten(arr: np.ndarray, shape: Tuple):
    return arr.reshape(arr.shape[0], *shape)


def take_first(dataset: TensorDataset, num_to_keep: int):
    return TensorDataset(dataset.tensors[0][0:num_to_keep], dataset.tensors[1][0:num_to_keep])


def center(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(0)
    return X_train - mean, X_test - mean


def standardize(X_train: np.ndarray, X_test: np.ndarray):
    std = X_train.std(0)
    return (X_train / std, X_test / std)


def _one_hot(tensor: Tensor, num_classes: int, default=0):
    M = F.one_hot(tensor, num_classes)
    M[M == 0] = default
    return M.float()


def make_labels(y, loss):
    if loss == "ce":
        return y
    elif loss == "mse":
        return _one_hot(y, 10, 0)


def load_cifar(loss: str) -> (TensorDataset, TensorDataset):
    cifar10_train = CIFAR10(root=DATASETS_FOLDER, download=True, train=True)
    cifar10_test = CIFAR10(root=DATASETS_FOLDER, download=True, train=False)
    X_train, X_test = flatten(cifar10_train.data / 255), flatten(cifar10_test.data / 255)
    y_train, y_test = make_labels(torch.tensor(cifar10_train.targets), loss), \
        make_labels(torch.tensor(cifar10_test.targets), loss)
    center_X_train, center_X_test = center(X_train, X_test)
    standardized_X_train, standardized_X_test = standardize(center_X_train, center_X_test)
    train = TensorDataset(torch.from_numpy(unflatten(standardized_X_train, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_train)
    test = TensorDataset(torch.from_numpy(unflatten(standardized_X_test, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_test)
    return train, test


def load_mnist(num_per_class=0, flatten=False, test=False):
    if test:
        with open(f'./data/mnist_test.npz', 'rb') as data_file:
            npzfile = np.load(data_file)
            X = npzfile['X'].astype(np.float32)
            Y = npzfile['Y']
    else:
        if num_per_class in [100, 300, 500, 1000, 2500]:
            tag = str(num_per_class)
        elif num_per_class == 0:
            tag = 'full'
        else:
            raise Exception('The specificed number of samples per class is not sampled.')
        with open(f'./data/mnist_train_{tag}.npz', 'rb') as data_file:
            npzfile = np.load(data_file)
            X = npzfile['X'].astype(np.float32)
            Y = npzfile['Y']
    if flatten:
        return torch.from_numpy(X.reshape(X.shape[0], -1) / 255.), torch.from_numpy(Y)
    else:
        return torch.from_numpy(X / 255.), torch.from_numpy(Y).to(torch.int64)
    
    
def load_reg_mnist(flatten=False, test=False):
    if test:
        with open(f'./data/mnist_test.npz', 'rb') as data_file:
            npzfile = np.load(data_file)
            X = npzfile['X'].astype(np.float32)
        with open(f'./data/mnist_test_reg_labels.npz', 'rb') as data_file:
            npzfile = np.load(data_file)
            Y = npzfile['Y'].astype(np.float32).reshape((-1, 1))
        
    else:
        with open(f'./data/mnist_train_500.npz', 'rb') as data_file:
            npzfile = np.load(data_file)
            X = npzfile['X'].astype(np.float32)
            
        with open(f'./data/mnist_train_500_reg_labels.npz', 'rb') as data_file:
            npzfile = np.load(data_file)
            Y = npzfile['Y'].astype(np.float32).reshape((-1, 1))
    if flatten:
        return torch.from_numpy(X.reshape(X.shape[0], -1) / 255.), torch.from_numpy(Y)
    else:
        return torch.from_numpy(X / 255.), torch.from_numpy(Y)


def load_dataset(dataset_name: str, loss: str) -> (TensorDataset, TensorDataset):
    if dataset_name == "cifar10":
        return load_cifar(loss)
    elif dataset_name == "cifar10-1k":
        train, test = load_cifar(loss)
        return take_first(train, 1000), test
    elif dataset_name == "cifar10-2k":
        train, test = load_cifar(loss)
        return take_first(train, 2000), test
    elif dataset_name == "cifar10-5k":
        train, test = load_cifar(loss)
        return take_first(train, 5000), test
    elif dataset_name == "cifar10-10k":
        train, test = load_cifar(loss)
        return take_first(train, 10000), test
    elif dataset_name == "cifar10-20k":
        train, test = load_cifar(loss)
        return take_first(train, 20000), test
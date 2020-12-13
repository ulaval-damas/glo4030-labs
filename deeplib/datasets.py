import os
import math
import random
import numpy as np
import torch
from torch.utils.data import Subset, Dataset, DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10

BASE_PATH = '~/GLO-4030/datasets/'


def load_mnist(path=os.path.join(BASE_PATH, 'mnist')):
    """
    Retourne l'ensemble d'entraînement du jeu de données MNIST. Le jeu de données est téléchargé s'il n'est pas présent.

    Args:
        path (str): Le répertoire où trouver ou télécharger MNIST.

    Returns:
        Tuple (jeu de données d'entraînement, jeu de données de test).
    """
    train_dataset = MNIST(path, train=True, download=True)
    test_dataset = MNIST(path, train=False, download=True)
    return train_dataset, test_dataset


def load_cifar10(path=os.path.join(BASE_PATH, 'cifar10')):
    """
    Retourne l'ensemble d'entraînement du jeu de données CIFAR10. Le jeu de données est téléchargé s'il n'est pas
    présent.

    Args:
        path (str): Le répertoire où trouver ou télécharger CIFAR10.

    Returns:
        Tuple (jeu de données d'entraînement, jeu de données de test).
    """
    train_dataset = CIFAR10(path, train=True, download=True)
    test_dataset = CIFAR10(path, train=False, download=True)
    return train_dataset, test_dataset


def train_valid_loaders(dataset, batch_size, train_split=0.8, shuffle=True, seed=42):
    """
    Divise un jeu de données en ensemble d'entraînement et de validation et retourne pour chacun un DataLoader PyTorch.

    Args:
        dataset (torch.utils.data.Dataset): Un jeu de données PyTorch
        batch_size (int): La taille de batch désirée pour le DataLoader
        train_split (float): Un nombre entre 0 et 1 correspondant à la proportion d'exemple de l'ensemble
            d'entraînement.
        shuffle (bool): Si les exemples sont mélangés aléatoirement avant de diviser le jeu de données.
        seed (int): Le seed aléatoire pour que l'ordre des exemples mélangés soit toujours le même.

    Returns:
        Tuple (DataLoader d'entraînement, DataLoader de test).
    """
    num_data = len(dataset)
    indices = np.arange(num_data)

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    split = math.floor(train_split * num_data)
    train_idx, valid_idx = indices[:split], indices[split:]

    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader


class SpiralDataset(Dataset):
    """
    Un jeu de données synthétique de spiral pour PyTorch.

    Args:
        n_points (int): Le nombre de point désiré dans le jeu de données
        noise (float): Quantité de bruit désiré dans le jeu de données
    """

    def __init__(self, n_points=1000, noise=0.2):
        self.points = torch.Tensor(n_points, 7)
        self.labels = torch.LongTensor(n_points)

        n_positive = n_points // 2
        n_negative = n_points = n_positive

        for i, point in enumerate(self._gen_spiral_points(n_positive, 0, noise)):
            self.points[i], self.labels[i] = point, 1

        for i, point in enumerate(self._gen_spiral_points(n_negative, math.pi, noise)):
            self.points[i + n_positive] = point
            self.labels[i + n_positive] = 0

    def _gen_spiral_points(self, n_points, delta_t, noise):
        for i in range(n_points):
            r = i / n_points * 5
            t = 1.75 * i / n_points * 2 * math.pi + delta_t
            x = r * math.sin(t) + random.uniform(-1, 1) * noise
            y = r * math.cos(t) + random.uniform(-1, 1) * noise
            yield torch.Tensor([x, y, x**2, y**2, x * y, math.sin(x), math.sin(y)])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.points[i], self.labels[i]

    def to_numpy(self):
        return self.points.numpy(), self.labels.numpy()

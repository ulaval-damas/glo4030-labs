from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10


def load_mnist():
    train_dataset = MNIST('dataset', train=True, download=True)
    test_dataset = MNIST('dataset', train=False, download=True)
    return train_dataset, test_dataset


def load_cifar10():
    train_dataset = CIFAR10('dataset', train=True, download=True)
    test_dataset = CIFAR10('dataset', train=False, download=True)
    return train_dataset, test_dataset

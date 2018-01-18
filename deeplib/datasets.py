from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10


def load_mnist():
    train_dataset = MNIST('/rap/colosse-users/GLO-4030/datasets/mnist', train=True, download=False)
    test_dataset = MNIST('/rap/colosse-users/GLO-4030/datasets/mnist', train=False, download=False)
    return train_dataset, test_dataset


def load_cifar10():
    train_dataset = CIFAR10('/rap/colosse-users/GLO-4030/datasets/cifar10', train=True, download=False)
    test_dataset = CIFAR10('/rap/colosse-users/GLO-4030/datasets/cifar10', train=False, download=False)
    return train_dataset, test_dataset

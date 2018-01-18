from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10


def load_mnist(download=False):
    train_dataset = MNIST('/rap/colosse-users/GLO-4030/datasets/mnist', train=True, download=download)
    test_dataset = MNIST('/rap/colosse-users/GLO-4030/datasets/mnist', train=False, download=download)
    return train_dataset, test_dataset


def load_cifar10(download=False):
    train_dataset = CIFAR10('/rap/colosse-users/GLO-4030/datasets/cifar10', train=True, download=download)
    test_dataset = CIFAR10('/rap/colosse-users/GLO-4030/datasets/cifar10', train=False, download=download)
    return train_dataset, test_dataset

if __name__ == '__main__':
    mnist = load_mnist(download=True)
    cifar = load_cifar10(download=True)

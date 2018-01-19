from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10


def load_mnist(download=False, path='/rap/colosse-users/GLO-4030/datasets/mnist'):
    train_dataset = MNIST(path, train=True, download=download)
    test_dataset = MNIST(path, train=False, download=download)
    return train_dataset, test_dataset


def load_cifar10(download=False, path='/rap/colosse-users/GLO-4030/datasets/cifar10'):
    train_dataset = CIFAR10(path, train=True, download=download)
    test_dataset = CIFAR10(path, train=False, download=download)
    return train_dataset, test_dataset

if __name__ == '__main__':
    mnist = load_mnist(download=True)
    cifar = load_cifar10(download=True)

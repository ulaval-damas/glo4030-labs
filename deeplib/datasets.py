import torch

from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10
from torch.autograd import Variable
from torchvision.transforms import ToTensor


class Dataset:

    def __init__(self, dataset):
        self.dataset = dataset
        self.index = 0

    def get_mini_batch(self, size, use_gpu=False):
        inputs = []
        targets = []
        for i in range(size):
            img, target = self.dataset[self.index]
            inputs.append(ToTensor()(img))
            targets.append(target)
            self.index = (self.index + 1) % len(self.dataset)

        inputs = Variable(torch.stack(inputs))
        targets = Variable(torch.LongTensor(targets))

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        return inputs, targets

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = 0 if key.start is None else key.start
            stop = len(self.dataset) if key.stop is None else key.stop
            step = 1 if key.step is None else key.step
            return [self.dataset[x] for x in range(start, stop, step)]
        return self.dataset[key]

    def __len__(self):
        return len(self.dataset)


def load_mnist(train=True):
    train_dataset = MNIST('dataset', train=train, download=True)
    dataset = Dataset(train_dataset)
    return dataset


def load_cifar10(train=True):
    train_dataset = CIFAR10('dataset', train=train, download=True)
    dataset = Dataset(train_dataset)
    return dataset

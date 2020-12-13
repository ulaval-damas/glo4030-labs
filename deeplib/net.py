# Adapted from
# http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py

import torch.nn as nn
import torch.nn.functional as F


class DeepMnistNet(nn.Module):
    # pylint: disable=too-many-instance-attributes
    """
    Un simple réseau profonds à convolution pour MNIST avec convolutions, relus et max-pooling.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1)
        self.conv2 = nn.Conv2d(10, 150, 3, padding=1)
        self.conv3 = nn.Conv2d(150, 300, 3, padding=1)
        self.conv4 = nn.Conv2d(300, 300, 3, padding=1)
        self.conv5 = nn.Conv2d(300, 150, 3, padding=1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(150 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MnistNet(nn.Module):
    """
    Un simple réseau moins profonds pour MNIST que DeepMnistNet à convolution avec convolutions, relus et max-pooling.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1)
        self.conv2 = nn.Conv2d(10, 50, 3, padding=1)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(50 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CifarNet(nn.Module):
    """
    Un simple réseau profonds à convolution pour CIFAR10 avec convolutions, relus et max-pooling.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 3, padding=1)
        self.conv2 = nn.Conv2d(10, 50, 3, padding=1)
        self.conv3 = nn.Conv2d(50, 150, 3, padding=1)
        self.fc1 = nn.Linear(150 * 8 * 8, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        x = self.fc1(x)
        return x


class CifarNetBatchNorm(nn.Module):
    """
    Un simple réseau profonds à convolution pour CIFAR10 avec batch normalization.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 5, padding=1)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 50, 3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(50)
        self.conv3 = nn.Conv2d(50, 150, 3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(150)
        self.fc1 = nn.Linear(2400, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), (2, 2))
        x = x.flatten(1)
        x = self.fc1(x)
        return x

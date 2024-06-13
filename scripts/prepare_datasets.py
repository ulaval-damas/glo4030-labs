import os

import deeplib.datasets as D

if __name__ == '__main__':
    D.load_mnist()
    D.load_cifar10()
    os.system(f'curl -o ~/GLO-4030/datasets/tiny_shakespeare.txt https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt')
    print('TODO upload COCO dataset to the cluster')

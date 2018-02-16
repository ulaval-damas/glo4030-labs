'''
From https://discuss.pytorch.org/t/print-autograd-graph/692/16
'''
from graphviz import Digraph
import torch
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.autograd import Variable
from torchvision.transforms import ToTensor


def show_worst(results):
    worst_results = []
    for i, result in enumerate(results):
        if len(worst_results) < 9:
            worst_results.append((result[1], i))

        else:
            if result[1] < worst_results[0][0]:
                worst_results[8] = (result[1], i)

        worst_results.sort()

    imgs, true, pred, score = [], [], [], []
    for i in range(9):
        worst = results[worst_results[i][1]]
        imgs.append(np.transpose(worst[0], (1, 2, 0)))
        score.append(worst[1])
        true.append(worst[2])
        pred.append(worst[3])

    imgs = np.asarray(imgs)
    plot_cifar_images(imgs, true, pred, score=score)


def show_best(results):
    best_results = []
    for i, result in enumerate(results):
        if len(best_results) < 9:
            best_results.append((result[1], i))

        else:
            if result[1] > best_results[0][0]:
                best_results[0] = (result[1], i)

        best_results.sort()

    imgs, true, pred, score = [], [], [], []
    for i in range(8, -1, -1):
        best = results[best_results[i][1]]
        imgs.append(np.transpose(best[0], (1, 2, 0)))
        score.append(best[1])
        true.append(best[2])
        pred.append(best[3])

    imgs = np.asarray(imgs)
    plot_cifar_images(imgs, true, pred, score=score)


def show_random(results):
    test = random.sample(results, 9)
    imgs, true, pred = [], [], []
    for i in range(9):
        imgs.append(np.transpose(test[i][0], (1, 2, 0)))
        true.append(test[i][2])
        pred.append(test[i][3])

    imgs = np.asarray(imgs)
    plot_cifar_images(imgs, true, pred)


def plot_cifar_images(images, cls_true, cls_pred=None, score=None):
    label_names = [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'
    ]
    plot_images(images, cls_true, label_names, cls_pred, score)


def plot_images(images, cls_true, label_names=None, cls_pred=None, score=None, gray=False):
    assert len(images) == len(cls_true) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        if gray:
            ax.imshow(images[i], cmap='gray', interpolation='spline16')
        else:
            ax.imshow(images[i, :, :, :], interpolation='spline16')
        # get its equivalent class name

        if label_names:
            cls_true_name = label_names[cls_true[i]]

            if cls_pred is None:
                xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
            elif score is None:
                cls_pred_name = label_names[cls_pred[i]]
                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
            else:
                cls_pred_name = label_names[cls_pred[i]]
                xlabel = "True: {0}\nPred: {1}\nScore: {2:.2f}%".format(cls_true_name, cls_pred_name, score[i] * 100)

            ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def make_vizualization_autograd(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph.

    Blue nodes are trainable Variables (weights, bias).
    Orange node are saved tensors for the backward pass.

    Args:
        var: output Variable
        params: list of (name, Parameters)
    """
    #param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d'% v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                # node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                node_name = '%s\n %s' % ("Var", size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(var)), str(id(u[0])))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)

    return dot


def view_filters(net, img):
    img = ToTensor()(img)
    img = Variable(img.unsqueeze(0), volatile=True)
    img = img.cuda()
    output = net.conv1(img)
    output = output.cpu().data.numpy()[0]

    fig, axes = plt.subplots(1, len(output))
    for i in range(len(output)):
        axes[i].imshow(output[i])
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    plt.show()
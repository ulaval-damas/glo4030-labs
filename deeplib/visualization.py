'''
From https://discuss.pytorch.org/t/print-autograd-graph/692/16
'''
import random
from graphviz import Digraph
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor


def show_worst(results):
    """
    Affiche les images de CIFAR10 qui induisent le réseau le plus en erreur, c'est-à-dire que les images dont la
    probabilité de prédiction de la vraie classe était parmi les plus basse.

    Args:
        results (List[Tuple]): Une liste de tuple telle que retournée par `deeplib.training.validate_ranking`.
    """
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
    """
    Affiche les images de CIFAR10 dont le réseau fait les meilleurs prédictions, c'est-à-dire que les images dont la
    probabilité de prédiction de la vraie classe était parmi les plus élevée.

    Args:
        results (List[Tuple]): Une liste de tuple telle que retournée par `deeplib.training.validate_ranking`.
    """
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
    """
    Affiche des images aléatoires de CIFAR10.

    Args:
        results (List[Tuple]): Une liste de tuple telle que retournée par `deeplib.training.validate_ranking`.
    """
    test = random.sample(results, 9)
    imgs, true, pred = [], [], []
    for i in range(9):
        imgs.append(np.transpose(test[i][0], (1, 2, 0)))
        true.append(test[i][2])
        pred.append(test[i][3])

    imgs = np.asarray(imgs)
    plot_cifar_images(imgs, true, pred)


def plot_cifar_images(images, cls_true, cls_pred=None, score=None):
    """
    Affiche une batch d'images de CIFAR10 avec différentes informations en fonction de ce qui est founir en argument.

    Args:
        images (np.ndarray): Une batch d'images de CIFAR10 sous la forme d'un array Numpy
        cls_true (list): Une liste contenant les classes respectives des images
        cls_pred (list): Une liste contenant les classes prédites des images
        score (list): Une liste contenant des probabilités des images (de n'importe quelle nature)
    """

    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plot_images(images, cls_true, label_names=label_names, cls_pred=cls_pred, score=score)


def plot_images(images, cls_true, *, label_names=None, cls_pred=None, score=None, gray=False):
    """
    Affiche une batch d'images avec différentes informations en fonction de ce qui est founir en argument.

    Args:
        images (np.ndarray): Une batch d'images sous la forme d'un array Numpy
        cls_true (list): Une liste contenant les classes respectives des images
        label_names (list): Une liste de string pour toutes les classes. L'index i de la liste devrait contenir le nom
            de la classe i.
        cls_pred (list): Une liste contenant les classes prédites des images
        score (list): Une liste contenant des probabilités des images (de n'importe quelle nature)
        gray (bool): Si c'est des images en teinte de gris.
    """
    assert len(images) == len(cls_true) == 9

    # Create figure with sub-plots.
    _, axes = plt.subplots(3, 3)

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


def make_vizualization_autograd(var):
    """ Produces Graphviz representation of PyTorch autograd graph.

    Blue nodes are trainable parameters (weights, bias).
    Orange node are saved tensors for the backward pass.

    Args:
        var: output tensor
    """
    node_attr = dict(style='filled', shape='box', align='left', fontsize='12', ranksep='0.1', height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + ', '.join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
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
    """
    Affiche le résultat des filtres à convolution sur une image. On suppose que le réseau a une couche `conv1` qui peut
    être appliqué sur la batch d'images passée en paramètre.

    Args:
        network (nn.Module): Un réseau de neurones PyTorch avec une couche `conv1`.
        img (Union[PILImage, torch.Tensor]): Une image.
    """
    with torch.no_grad():
        if not torch.is_tensor(img):
            img = ToTensor()(img)
        img = img.unsqueeze(0)
        img = img.cuda()
        output = net.conv1(img)
        output = output.cpu().numpy()[0]

    _, axes = plt.subplots(1, len(output))
    for i, out in enumerate(output):
        axes[i].imshow(out)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    plt.show()


def show_2d_function(fct, min_val=-5, max_val=5, mesh_step=.01, *, optimal=None, bar=True, ax=None, **kwargs):
    # pylint: disable=blacklisted-name
    """
    Trace les courbes de niveau d'une fonction 2D.

    Args:
        fct (Callable[torch.Tensor, torch.Tensor]): Fonction objectif qui prend en paramètre un tenseur Nx2
            correspondant à N paramètres pour lesquels on veut obtenir la valeur de la fonction.
        optimal (torch.Tensor): La valeur optimale des poids pour la fonction objectif.
    """
    w1_values = torch.arange(min_val, max_val + mesh_step, mesh_step)
    w2_values = torch.arange(min_val, max_val + mesh_step, mesh_step)

    w2, w1 = torch.meshgrid(w2_values, w1_values)
    w_grid = torch.stack((w1.flatten(), w2.flatten()))
    fct_values = fct(w_grid).view(w1_values.shape[0], w2.shape[0]).numpy()

    w1_values, w2_values = w1_values.numpy(), w2_values.numpy()

    if ax is not None:
        plt.sca(ax)
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'RdBu'
    plt.contour(w1_values, w2_values, fct_values, 40, **kwargs)
    plt.xlim((min_val, max_val))
    plt.ylim((min_val, max_val))
    plt.xlabel('$w_1$')
    plt.ylabel('$w_2$')

    if bar:
        plt.colorbar()

    if optimal is not None:
        plt.scatter(*optimal.numpy(), s=200, marker='*', c='r')


def show_2d_trajectory(w_history, fct, min_val=-5, max_val=5, mesh_step=.5, *, optimal=None, ax=None):
    """
    Trace le graphique de la trajectoire de descente en gradient en 2D.

    Args:
        w_history: L'historique de la valeur des poids lors de l'entraînement.
        fct (Callable[torch.Tensor, torch.Tensor]): Fonction objectif qui prend en paramètre un tenseur Nx2
            correspondant à N paramètres pour lesquels on veut obtenir la valeur de la fonction.
        optimal (torch.Tensor): La valeur optimale des poids pour la fonction objectif.
    """
    show_2d_function(fct, min_val, max_val, mesh_step, optimal=optimal, ax=ax)

    if len(w_history) > 0:
        trajectory = np.array(w_history)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'o--', c='g')

    plt.title('Trajectoire de la descente en gradient')
    plt.xlabel('$w_1$')
    plt.ylabel('$w_2$')


def show_learning_curve(loss_list, loss_opt=None, ax=None):
    """
    Trace le graphique des valeurs de la fonction objectif lors de l'apprentissage.

    Args:
        loss_list: L'historique de la valeur de la perte lors de l'entraînement.
        loss_opt: La valeur optimale de perte.
    """
    if ax is not None:
        plt.sca(ax)
    plt.plot(np.arange(1, len(loss_list) + 1), loss_list, 'o--', c='g', label='$F(\\mathbf{w})$')
    if loss_opt is not None:
        plt.plot([1, len(loss_list)], 2 * [loss_opt], '*--', c='r', label='optimal')
    plt.title('Valeurs de la fonction objectif')
    plt.xlabel('Itérations')
    plt.legend()


def show_optimization(w_history, loss_history, fct, optimal=None, title=None):
    """
    Trace deux graphiques montrant le trajet de l'optimisation d'une fonction objectif 2D. Le premier montre la valeur
    des poids lors de l'optimisation. Le deuxième montre la valeur de la perte lors de l'optimisation.

    Args:
        w_history: L'historique des poids lors de l'optimisation
        loss_history: L'historique de la valeur de la fonction perte.
        fct (Callable[torch.Tensor, torch.Tensor]): Fonction objectif qui prend en paramètre un tenseur Nx2
            correspondant à N paramètres pour lesquels on veut obtenir la valeur de la fonction.
        optimal (torch.Tensor): La valeur optimale des poids pour la fonction objectif.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4))
    if title is not None:
        fig.suptitle(title)
    show_2d_trajectory(w_history, fct, optimal=optimal, ax=axes[0])
    show_learning_curve(loss_history, loss_opt=fct(optimal).numpy(), ax=axes[1])

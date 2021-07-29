import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import poutyne as pt

from deeplib.history import History
from deeplib.datasets import train_valid_loaders


def get_model(network, optimizer=None, criterion=None, use_gpu=True, acc=True):
    """
    Obtient un modèle Poutyne pour un réseau de neurones PyTorch. On suppose que la sortie du réseau est compatible avec
    la fonction cross-entropy de PyTorch pour pouvoir utiliser l'exactitude (accuracy).

    Args:
        network (nn.Module): Un réseau de neurones PyTorch
        optimizer (torch.optim.Optimizer): Un optimiseur PyTorch
        criterion: Une fonction de perte compatible avec la cross-entropy de PyTorch
        use_gpu (bool): Si on veut utiliser le GPU. Est vrai par défaut. Un avertissement est lancé s'il n'y a pas de
            GPU.
        acc (bool): Si on veut inclure l'exactitude (accuracy) comme métrique à calculer.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    batch_metrics = ['accuracy'] if acc else []
    model = pt.Model(network, optimizer, criterion, batch_metrics=batch_metrics)
    if use_gpu:
        if torch.cuda.is_available():
            model.cuda()
        else:
            warnings.warn("Aucun GPU disponible")
    return model


def softmax(x, axis=1):
    """
    Implémente la fonction softmax avec NumPy de manière numériquement stable.

    Args:
        x (np.ndarray): Le array NumPy à appliquer la softmax
        axis (int): L'axe sur lequel appliqué la softmax. Par défaut, c'est l'axe 1 étant donné que la cross-entropy
            dans PyTorch suppose que les logits sont sur l'axe 1.
    """
    e_x = np.exp(x - x.max(axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def validate_ranking(network, dataset, batch_size, use_gpu=True):
    """
    Sépare les exemples d'un jeu de données en deux listes: une liste d'exemple bien classifié et une liste d'exemple
    mal classifié. On suppose que la sortie du réseau est compatible avec la fonction cross-entropy de PyTorch pour
    calculer les prédictions.

    Args:
        network (nn.Module): Un réseau de neurones PyTorch
        dataset (torch.utils.data.Dataset): Un jeu de données PyTorch
        batch_size (int): La taille de batch désirée pour l'inférence
        use_gpu (bool): Si on veut utiliser le GPU. Est vrai par défaut. Un avertissement est lancé s'il n'y a pas de
            GPU.

    Returns:
        Retourne deux listes: une liste d'exemple bien classifié et une liste d'exemple mal classifié. Les éléments des
        deux listes sont des tuples sous format `(exemple, probabilité, classe, prédiction)`, où `exemple` est l'exemple
        (ou l'image) que l'on souhaite classifier, `probabilité` est à la probabilité donnée à la vraie classe de
        l'image, `classe` est la vraie classe de l'image et `prédiction` est la prédiction donnée par le réseau.

    See:
        Voir dans le module `deeplib.visualization` les fonction `show_best`, `show_worst` et `show_random`.
    """
    # pylint: disable=too-many-locals
    dataset.transform = ToTensor()
    loader = DataLoader(dataset, batch_size=batch_size)

    good = []
    errors = []

    model = get_model(network, use_gpu=use_gpu)
    for inputs, targets in loader:
        outputs = model.predict_on_batch(inputs)
        probs = softmax(outputs)
        predictions = outputs.argmax(1)

        inputs = inputs.cpu().numpy()
        targets = targets.cpu().numpy()
        for i, input_ in enumerate(inputs):
            score = probs[i][targets[i]]
            target = targets[i]
            pred = predictions[i]
            if target == pred:
                good.append((input_, score, target, pred))
            else:
                errors.append((input_, score, target, pred))

    return good, errors


class HistoryCallback(pt.Callback):
    """
    Un callback Poutyne pour sauvegarder le taux d'apprentissaage dans un objet de type `deeplib.history.History` en
    plus des autres métriques d'entraînement retourné par `Model.fit_generator`.

    Attributes:
        history (deeplib.history.History): L'objet d'historique de deeplib.
    """

    def __init__(self):
        super().__init__()
        self.history = History()

    def on_epoch_end(self, epoch_number, logs):
        self.history.save(dict(**logs, lr=self.model.optimizer.param_groups[0]['lr']))


def train(network, optimizer, dataset, n_epoch, batch_size, *, use_gpu=True, criterion=None, callbacks=None, acc=True):
    """
    Entraîne un réseau de neurones PyTorch avec Poutyne. On suppose que la sortie du réseau est compatible avec
    la fonction cross-entropy de PyTorch pour calculer l'exactitude (accuracy).

    Args:
        network (nn.Module): Un réseau de neurones PyTorch
        optimizer (torch.optim.Optimizer): Un optimiseur PyTorch
        dataset (torch.utils.data.Dataset): Un jeu de données PyTorch
        n_epoch (int): Le nombre d'epochs d'entraînement désiré
        batch_size (int): La taille de batch désirée
        use_gpu (bool): Si on veut utiliser le GPU. Est vrai par défaut. Un avertissement est lancé s'il n'y a pas de
            GPU.
        criterion: Une fonction de perte compatible avec la cross-entropy de PyTorch.
        callbacks (List[poutyne.Callback]): Une liste de callbacks de Poutyne (utile pour les horaires d'entrainement
            entre autres).
        acc (bool): Si on veut inclure l'exactitude (accuracy) comme métrique à calculer.

    Returns:
        Retourne un objet de type `deeplib.history.History` contenant l'historique d'entraînement.
    """
    history_callback = HistoryCallback()
    callbacks = [history_callback] if callbacks is None else [history_callback] + callbacks

    dataset.transform = ToTensor()
    train_loader, valid_loader = train_valid_loaders(dataset, batch_size=batch_size)

    model = get_model(network, optimizer, criterion, use_gpu=use_gpu, acc=acc)
    model.fit_generator(train_loader, valid_loader, epochs=n_epoch, callbacks=callbacks)

    return history_callback.history


def test(network, test_dataset, batch_size, *, use_gpu=True, criterion=None, acc=True):
    """
    Test un réseau de neurones PyTorch avec Poutyne. On suppose que la sortie du réseau est compatible avec
    la fonction cross-entropy de PyTorch pour calculer l'exactitude (accuracy).

    Args:
        network (nn.Module): Un réseau de neurones PyTorch
        test_dataset (torch.utils.data.Dataset): Un jeu de données PyTorch
        batch_size (int): La taille de batch désirée
        use_gpu (bool): Si on veut utiliser le GPU. Est vrai par défaut. Un avertissement est lancé s'il n'y a pas de
            GPU.
        criterion: Une fonction de perte compatible avec la cross-entropy de PyTorch

    Returns:
        Retourne l'exactitude sur le dataset.
    """
    test_dataset.transform = ToTensor()
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = get_model(network, criterion=criterion, use_gpu=use_gpu, acc=acc)
    metrics = model.evaluate_generator(test_loader)

    return metrics[1] if acc else metrics

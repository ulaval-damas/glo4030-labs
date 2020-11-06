import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import poutyne as pt

from deeplib.history import History
from deeplib.datasets import train_valid_loaders


def get_model(network, optimizer=None, criterion=None, use_gpu=True):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model = pt.Model(network, optimizer, criterion, batch_metrics=['accuracy'])
    if use_gpu:
        if torch.cuda.is_available():
            model.cuda()
        else:
            warnings.warn("Aucun GPU disponible")
    return model


def softmax(x, axis=1):
    e_x = np.exp(x - x.max(axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def validate_ranking(network, dataset, batch_size, use_gpu=True, criterion=None):
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
        for i in range(len(inputs)):
            score = probs[i][targets[i]]
            target = targets[i]
            pred = predictions[i]
            if target == pred:
                good.append((inputs[i], score, target, pred))
            else:
                errors.append((inputs[i], score, target, pred))

    return good, errors


class HistoryCallback(pt.Callback):
    def __init__(self):
        super().__init__()
        self.history = History()

    def on_epoch_end(self, epoch_number, logs):
        self.history.save(logs['acc'], logs['val_acc'],
                          logs['loss'], logs['val_loss'],
                          self.model.optimizer.param_groups[0]['lr'])


def train(network, optimizer, dataset, n_epoch, batch_size, use_gpu=True, criterion=None, callbacks=None):
    history_callback = HistoryCallback()
    callbacks = [history_callback] if callbacks is None else [history_callback] + callbacks

    dataset.transform = ToTensor()
    train_loader, valid_loader = train_valid_loaders(dataset, batch_size=batch_size)

    model = get_model(network, optimizer, criterion, use_gpu=use_gpu)
    model.fit_generator(train_loader, valid_loader,
                        epochs=n_epoch,
                        callbacks=callbacks,
                        progress_options=dict(coloring={
                            "text_color": 'MAGENTA',
                            "ratio_color": "GREEN",
                            "metric_value_color": "LIGHTBLUE_EX",
                            "time_color": "CYAN",
                            "progress_bar_color": "MAGENTA"
                        }))

    return history_callback.history


def test(network, test_dataset, batch_size, use_gpu=True, criterion=None):
    test_dataset.transform = ToTensor()
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = get_model(network, criterion=criterion, use_gpu=use_gpu)
    loss, acc = model.evaluate_generator(test_loader)

    return acc

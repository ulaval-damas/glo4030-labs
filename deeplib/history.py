from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class History:
    """
    Classe utilitaire pour enregistrer et faire des graphiques pour les métriques d'entraînement.

    Args:
        logs_list (List[Dict], optionnel): Un liste de dictionnaire contenant les différentes métrique pour chaque
            epoch. Le dictionnaire devrait avoir les mêmes clés que prises par la méthode `save()`.
    """

    def __init__(self, logs_list=None):
        self.history = defaultdict(list)

        if logs_list is not None:
            for logs in logs_list:
                self.save(logs)

    def save(self, logs):
        """
        Enregistre différentes métrique dans l'historique.

        Args:
            logs (dict): Le dictionnaire de métriques. Dans cette classe, on suppose que les clés suivantes sont dans le
                dictionnaire: loss, acc, val_loss, val_acc. Également, la clé 'lr' peut être présente.
        """
        for k, v in logs.items():
            self.history[k].append(v)

    def display(self, display_lr=False):
        """
        Affiche deux graphiques pour les pertes en entraînement et en validation et pour l'exactitude (accuracy) en
        entraînement et en validation. De plus, si le taux d'apprentissage a été sauvegardé, il est possible de lui
        afficher un graphique.

        Args:
            display_lr (bool): Si on veut afficher un graphique pour le taux d'apprentissage. Par défaut, le graphique
                n'est pas affiché.
        """
        epoch = len(self.history['loss'])
        epochs = list(range(1, epoch + 1))

        num_plots = 3 if display_lr else 2
        _, axes = plt.subplots(num_plots, 1, sharex=True)
        plt.tight_layout()

        axes[0].set_ylabel('Accuracy')
        axes[0].plot(epochs, self.history['acc'], label='Train')
        axes[0].plot(epochs, self.history['val_acc'], label='Validation')
        axes[0].legend()

        axes[1].set_ylabel('Loss')
        axes[1].plot(epochs, self.history['loss'], label='Train')
        axes[1].plot(epochs, self.history['val_loss'], label='Validation')

        if display_lr and 'lr' in self.history:
            axes[2].set_xlabel('Epochs')
            axes[2].set_ylabel('Lr')
            axes[2].plot(epochs, self.history['lr'], label='Lr')
            axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            axes[1].set_xlabel('Epochs')
            axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.show()

    def display_loss(self):
        """
        Affiche le graphique pour les pertes en entraînement et en validation

        """
        epoch = len(self.history['loss'])
        epochs = list(range(1, epoch + 1))

        plt.tight_layout()

        plt.plot(epochs, self.history['loss'], label='Train')
        plt.plot(epochs, self.history['val_loss'], label='Validation')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.show()

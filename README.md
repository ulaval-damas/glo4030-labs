# glo4030-labs

Laboratoires du cours GLO-4030/GLO-7030

## Instructions

### Exécution sur Hélios avec JupyterHub

La manière la plus simple de travailler sur les laboratoires est de passer par
le JupyterHub de Calcul Québec. Par contre, il faut comprendre que pour avoir
accès à une machine avec GPU, il faut se mettre en file d'attente dans le *batch
system* pour se faire allouer des ressources. Cela devrait prendre moins d'une
minute la plupart du temps. Voici les étapes à suivre:

1. Vous rendre au https://jupyter.calculquebec.ca/hub/login
2. Vous connecter avec votre compte Calcul Canada
3. Remplir le formulaire avec les informations suivantes:
    - Number of cores: entre 1 et 4
    - Memory: au moins 8192
    - GPU configuration: 1 x K80 **Important! Les cartes K20 ne fonctionnent plus 
      avec les versions récentes de PyTorch**
    - Reservation: GLO7030 sur les heures réservées, sinon None
    - Garder les choix par défaut pour le reste des options
4. Appuyer sur le bouton *Spawn*


Une fois connecté, vous devriez avoir accès au système de fichier. Le répertoire
du cours se situe au `/project/glo7030/`. Il contient les jeux de
données, les laboratoires et l'environnement virtuel python.


> **IMPORTANT**
> Vous n'avez accès qu'en lecture au notebook des laboratoires dans le répertoire
> du cours. Avant de travailler sur un laboratoire, veillez le copier dans votre
> propre répertoire.


Les prochaines étapes se font en ligne de commande directement dans Jupyter:

1. Ouvrir un terminal en cliquant sur New > Terminal
2. `source /project/glo7030/venv/bin/activate`
3. Créer un kernel Jupyer de l'environnement (cela permet de lancer des
   notebooks dans l'environnement virtuel): `python -m ipykernel install
   --user --name glo4030-7030`
4. Faire un lien symbolique pour plus rapidement accéder aux fichiers du cours
   avec `ln -s /project/glo7030/ ~/GLO-4030`. Le répertoire du cours est
   maintenant en raccourci dans votre $HOME.
5. Copier le labo 1 dans votre $HOME `cp ~/GLO-4030/glo4030-labs/Laboratoire\ 1.ipynb ~/`. Cela
   vous permet de sauvegarder vos résultats et modifications. Vous n'avez accès
   qu'en lecture seule aux fichiers du répertoire du cours. Vous aurez à répéter cette opération
   lors de chaque laboratoire.
6. Quitter la console avec CTRL-D puis fermez la fenêtre.


À cette étape, vous devriez avoir un kernel Jupyter fonctionnel. Dans la liste
des kernels dans *New*, il devrait y avoir *glo4030-7030*. S'il n'apparaît pas,
rafraîchir la page.

Pour démarrer un laboratoire, double-cliquer sur le laboratoire. Une fois ouvert, faire
Kernel > Change Kernel > glo4030-7030. Vous êtes maintenant prêt à commencer le laboratoire.


### Exécution sur Google Colab

Il est possible d'utiliser Google Colab pour les laboratoires suivants:

- [Laboratoire 1](https://colab.research.google.com/github/ulaval-damas/glo4030-labs/blob/master/Laboratoire%201.ipynb)
- [Laboratoire 2](https://colab.research.google.com/github/ulaval-damas/glo4030-labs/blob/master/Laboratoire%202.ipynb)
- [Laboratoire 3](https://colab.research.google.com/github/ulaval-damas/glo4030-labs/blob/master/Laboratoire%203.ipynb)
- [Laboratoire 4](https://colab.research.google.com/github/ulaval-damas/glo4030-labs/blob/master/Laboratoire%204.ipynb)
- [Laboratoire 5](https://colab.research.google.com/github/ulaval-damas/glo4030-labs/blob/master/Laboratoire%205.ipynb)
- [Laboratoire 6](https://colab.research.google.com/github/ulaval-damas/glo4030-labs/blob/master/Laboratoire%206.ipynb)
- [Laboratoire 7](https://colab.research.google.com/github/ulaval-damas/glo4030-labs/blob/master/Laboratoire%207.ipynb)

Le seul prérequis est d'avoir un compte Google.


### Installation locale

Il est possible d'installer en local les laboratoires. Il vous faudra idéalement
une machine avec GPU. Les dépendances sont les suivantes:

- pytorch: Voir le site web (http://pytorch.org/) pour plus détails concernant l'installation)
- les dépendances du fichier `requirements.txt` (avec `pip install -r requirements.txt`)

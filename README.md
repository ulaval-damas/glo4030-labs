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
2. Vous connecter avec votre compte Calcul Québec
3. Appuyer sur le bouton *Start my server*
4. Indiquer la durée de votre réservation de machine avec GPU dans *Runtime* et
   cocher *Require a GPU* et *Enable Compte Canada software stack*


Une fois connecté, vous devriez avoir accès au système de fichier. Dans votre
*home* se trouve le répertoire du cours. Il contient les jeux de données, les
laboratoires et l'environnement virtuel python. 


> **IMPORTANT**
> Vous n'avez accès qu'en lecture au notebook des laboratoires dans le répertoire
> du cours. Avant de travailler sur un laboratoire, veillez le copier dans votre
> propre répertoire.


Les prochaines étapes se font en ligne de commande directement dans Jupyter:

1. Ouvrir un terminal en cliquant sur New > Terminal
2. Copier l'environnement virtuel avec `cp -r glo4030-7030/venv ~/venv`
3. `module load apps/python/3.5.0`
4. `source ~/venv/bin/activate`
5. Création d'un kernel Jupyer de votre environnement (cela permet de lancer des
   notebooks dans votre environnement virtuel): `python -m ipykernel install
   --user --name glo4030-7030`
6. Copie du labo 1 en local `cp glo4030-7030/labs/Laboratoire1.ipynb ~/`
7. Quitter la console avec CTRL-D puis fermez la fenêtre.


À cette étape, vous devriez avoir un kernel Jupyter fonctionnel. Dans la liste
des kernels dans *New*, il devrait y avoir *glo4030-7030*.


### Installation locale

Il est possible d'installer en local les laboratoires. Il vous faudra idéalement
une machine avec GPU. Les dépendances sont les suivantes:

- cuda
- cudnn
- pytorch http://pytorch.org/
- les dépendances du fichier `requirements.txt`

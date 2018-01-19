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


Une fois connecté, vous devriez avoir accès au système de fichier. Le répertoire
du cours se situe au `/rap/colosse-users/GLO-4030`. Il contient les jeux de
données, les laboratoires et l'environnement virtuel python.


> **IMPORTANT**
> Vous n'avez accès qu'en lecture au notebook des laboratoires dans le répertoire
> du cours. Avant de travailler sur un laboratoire, veillez le copier dans votre
> propre répertoire.


Les prochaines étapes se font en ligne de commande directement dans Jupyter:

1. Ouvrir un terminal en cliquant sur New > Terminal
2. `source /rap/colosse-users/GLO-4030/venv/bin/activate`
3. Créer un kernel Jupyer de l'environnement (cela permet de lancer des
   notebooks dans l'environnement virtuel): `python -m ipykernel install
   --user --name glo4030-7030`
4. Faire un lien symbolique pour plus rapidement accéder aux fichiers du cours
   avec `ln -s /rap/colosse-users/GLO-4030 ~`. Le répertoire du cours est
   maintenant en raccourci dans votre $HOME.
5. Aller dans l'onglet Softwares de JupyterHub et ajoutez `cuda 8` et `cudnn 7`.
   Appuyer sur `Save` sans préciser de nom de collection; cela modifie les modules par défaut.
6. Copier le labo 1 dans votre $HOME `cp ~/glo4030-7030/labs/Laboratoire\ 1.ipynb ~/`. Cela
   vous permet de sauvegarder vos résultats et modifications. Vous n'avez accès
   qu'en lecture seule aux fichiers du répertoire du cours.
7. Quitter la console avec CTRL-D puis fermez la fenêtre.


À cette étape, vous devriez avoir un kernel Jupyter fonctionnel. Dans la liste
des kernels dans *New*, il devrait y avoir *glo4030-7030*.


### Installation locale

Il est possible d'installer en local les laboratoires. Il vous faudra idéalement
une machine avec GPU. Les dépendances sont les suivantes:

- cuda
- cudnn
- pytorch http://pytorch.org/
- les dépendances du fichier `requirements.txt` (avec `pip install -r requirements.txt`)

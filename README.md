# glo4030-labs

Laboratoires du cours GLO-4030/GLO-7030

## Usage de l'Infrastructure de calcul pour l'enseignement (ICE)

La manière recommandée de travailler sur les laboratoires est de passer par ICE. Cette plateforme sera également disponible pour les travaux pratiques et le projet de session.
Vous pouvez trouver la documentation de la plateforme [sur cette page](https://ul-ice-docs.s3.valeria.science/index.html).

### Exécution sur JupyterHub

Voici les étapes initiales à suivre:
1. Si ce n'est pas déjà fait, vous connecter au réseau ou au VPN de l'Université Laval
2. Vous rendre sur https://jupyter.ice.ulaval.ca/hub/home
3. Vous connecter avec votre compte ULaval
4. Sélectionner *Démarrer mon serveur*
5. Voir [cette section](https://ul-ice-docs.s3.valeria.science/index.html#4-utilisation-du-calcul-gpu-avec-jupyter) de la documentation pour les paramètres à choisir
6. Appuyer sur le bouton *Démarrer*
7. **La mise en place n'est pas encore terminée! Continuez à lire ce document!**

Une fois connecté, vous devriez avoir accès au système de fichier. Le répertoire
du cours se situe au `/public/enseignement/GLO-4030`. Il contient les jeux de
données, les laboratoires et l'environnement virtuel python.

> **IMPORTANT**
> Vous n'avez accès qu'en lecture au répertoire. Avant de travailler sur un laboratoire, suivez les prochaines étapes.

1. Ouvrir un terminal en cliquant sur `File > New > Terminal`
2. Faire un lien symbolique vers les fichiers du cours avec `ln -s /public/enseignement/GLO-4030 ~/GLO-4030`.
3. Faire un lien symbolique vers deeplib avec `ln -s ~/GLO-4030/code/glo4030-labs/deeplib ~/deeplib`

Le répertoire du cours est maintenant en raccourci dans votre dossier personnel, mais il est encore en lecture seule. Au début de chaque laboratoire, vous devrez:

1. Charger le module du cours en suivant [cette section](https://ul-ice-docs.s3.valeria.science/index.html#etape-3-chargement-du-module-personnalise-pour-le-cours) de la documentation.
   Cette étape est cruciale pour accéder à l'environnement virtuel des laboratoires.
2. Copier le notebook du laboratoire du jour dans votre $HOME, e.g. `cp ~/GLO-4030/code/glo4030-labs/Laboratoire\ 1.ipynb ~/`.
   Cela vous permet de sauvegarder vos résultats et modifications.
3. Pour démarrer le laboratoire, double-cliquer sur le notebook dans l'arborescence de fichier à gauche. Si vous ne voyez pas le fichier, rafraîchissez la page.

### Lancer des `jobs` sur l'Infrastructure de calcul pour l'enseignement (ICE) avec Slurm

Cette approche sera plutôt pertinente pour les travaux pratiques et le projet de session.
Voir [cette section](https://ul-ice-docs.s3.valeria.science/index.html#5-utilisation-du-calcul-gpu-avec-slurm-mode-batch) de la documentation.

## Exécution sur Google Colab

Il est possible d'utiliser Google Colab pour les laboratoires. Le seul prérequis est d'avoir un compte Google.
Nous recommandons toutefois d'utiliser la plateforme ICE, qui donne accès à de meilleures ressources computationnelles.
Vous pouvez accéder aux laboratoires avec Google Colab à partir des liens suivants:

- [Laboratoire 1](https://colab.research.google.com/github/ulaval-damas/glo4030-labs/blob/master/Laboratoire%201.ipynb)
- [Laboratoire 2](https://colab.research.google.com/github/ulaval-damas/glo4030-labs/blob/master/Laboratoire%202.ipynb)
- [Laboratoire 3](https://colab.research.google.com/github/ulaval-damas/glo4030-labs/blob/master/Laboratoire%203.ipynb)
- [Laboratoire 4](https://colab.research.google.com/github/ulaval-damas/glo4030-labs/blob/master/Laboratoire%204.ipynb)
- [Laboratoire 5](https://colab.research.google.com/github/ulaval-damas/glo4030-labs/blob/master/Laboratoire%205.ipynb)
- [Laboratoire 6](https://colab.research.google.com/github/ulaval-damas/glo4030-labs/blob/master/Laboratoire%206.ipynb)
- [Laboratoire 7](https://colab.research.google.com/github/ulaval-damas/glo4030-labs/blob/master/Laboratoire%207.ipynb)
- [Laboratoire 8](https://colab.research.google.com/github/ulaval-damas/glo4030-labs/blob/master/Laboratoire%208.ipynb)

## Installation locale

Il est possible d'exécuter les laboratoires à partir d'une installation locale. Il vous faudra idéalement
une machine avec GPU. Les dépendances sont les suivantes:

- pytorch: Voir le site web (http://pytorch.org/) pour plus détails concernant l'installation)
- les dépendances du fichier `requirements.txt` (avec `pip install -r requirements.txt`)
- `sudo apt install -y graphviz`
- intaller JupyterLab `pip install jupyter`
- lancer JupyterLab `jupyter lab`

## Utilisation de devcontainer pour `vscode`

Nous fournissons un fichier `.devcontainer/devcontainer.json` afin de permettre le développement dans Visual Studio Code à l'intérieur d'un conteneur Docker.
Pour plus de détails sur le développement dans un container, vous référer à [Developing inside a Container](https://code.visualstudio.com/docs/devcontainers/containers).

> **IMPORTANT**
> Vous devrez peut-être modifier les versions de drivers utilisés dans `Dockerfile`.
> 
> Line 1: `nvidia/cuda:11.4.3-base-ubuntu20.04`
> Line 26: `RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
> 

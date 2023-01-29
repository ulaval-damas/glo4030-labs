# Exécution de `jobs` sur la grappe de Calcul Québec

1. Connexion par SSH à l'URL suivante: `glo4030.calculquebec.cloud`.

```bash
# remplacez <username> par votre notre d'utilisateur
ssh <username>@glo4030.calculquebec.cloud
```

2. Déposer votre code sur le serveur avec `git clone`

3. Création de l'environnement Python.

```bash
python -m venv venv
source venv/bin/activate
# pip install -r requirements.txt
```

4. Déposer des données sur la grappe avec [FileZilla](https://filezilla-project.org/). Vous devrez vous connecter à l'adresse `glo4030.calculquebec.cloud` avec votre nom d'utilisateur et votre mot de passe.

5. Lancer une `job`

Créer un script `job.sh` avec le contenu suivant:

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8192M
#SBATCH --time=0-12:00
#SBATCH --job-name=train_network
#SBATCH --output=%x-%j.out

# Variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MAKEFLAGS="-j$(nproc)"

# Setup Python
source ~/venv/bin/activate

# Start task
# TODO lancer votre script ici
# python main.py

# Cleaning up
deactivate
```

Puis, lancez la `job` avec

```bash
sbatch job.sh
```

6. Résultats

Utilisez `sq` pour voir l'état de vos jobs. La sortie des jobs sera écrite dans le fichier `job-XXX.out`, où `XXX` est le numéro de la job.

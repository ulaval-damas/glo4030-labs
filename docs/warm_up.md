# Warm-up des GPUs sur Calcul Québec

Cette section est réservée n'est pertinente que pour les auxiliaires d'enseignement.

Environ 30 minutes avant le début des laboratoires, il est nécessaire de lancer des jobs pour préparer la grappe de calculs.

Créer une job dans le fichier `allocate_gpu.sh`.

```shell
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:30
#SBATCH --job-name=warm_up_gpu
#SBATCH --output=%x-%j.out

echo "Warming up"
sleep 1210
echo "Exiting"
```

Créer un script qui va lancer plusieurs jobs, `allocate_all_gpu.sh`

```shell
for i in $(seq 1 30);
do
    sbatch allocate_gpu.sh
done
```

Pour annuler toutes les jobs, `cancel_all.sh`:

```shell
squeue | awk 'NR>1 {print $1}' | xargs scancel
rm *.out
```

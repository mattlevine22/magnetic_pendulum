#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=50GB
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err

base_dir="/groups/astuart/mlevine/magnetic_pendulum/results"

srun --ntasks=1 python script.py --n_random_features 3000 --n_per_axis 50 --base_dir $base_dir &
srun --ntasks=1 python script.py --n_random_features 1500 --n_per_axis 50 --base_dir $base_dir &
srun --ntasks=1 python script.py --n_random_features 3000 --n_per_axis 25 --base_dir $base_dir &
srun --ntasks=1 python script.py --n_random_features 1500 --n_per_axis 25 --base_dir $base_dir &
srun --ntasks=1 python script.py --n_random_features 1000 --n_per_axis 100 --base_dir $base_dir &
srun --ntasks=1 python script.py --n_random_features 1000 --n_per_axis 75 --base_dir $base_dir &
srun --ntasks=1 python script.py --n_random_features 1000 --n_per_axis 50 --base_dir $base_dir &
srun --ntasks=1 python script.py --n_random_features 1000 --n_per_axis 25 --base_dir $base_dir
wait

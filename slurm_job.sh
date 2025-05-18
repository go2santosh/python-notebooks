#!/bin/bash
#SBATCH --job-name='image-translation-cycle-gan'
#SBATCH --chdir=/data/home/ss231/image-to-image-translation-gan
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --time=30-00:00
#SBATCH --mincpus=8

module load python/3.12

conda activate /data/home/ss231/conda_env
python3.12 image-translation-cycle-gan.py
deactivate
        
echo job end time is `date`
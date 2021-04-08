#!/bin/bash

#SBATCH --partition=normal
#SBATCH --ntasks=1
# memory in MB
#SBATCH --mem=1024
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=results_hw5/multi_image_inception_%04a.txt
#SBATCH --error=results_hw5/error_MII_%04a.txt
#SBATCH --time=12:00:00
#SBATCH --job-name=multi_image_inception
#SBATCH --mail-user=michael.montalbano@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/mcmontalbano/HW5
#SBATCH --array=0-5
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
source ~fagg/pythonenv/tensorflow/bin/activate

python hw5_multiple_inputs.py -exp_index $SLURM_ARRAY_TASK_ID -epochs 2000 -patience 1000 -L2_regularizer 0.001 -experiment_type 'basic' -dataset '/home/fagg/datasets/core50/core50_128x128' -network 'inception'
#!/bin/bash

#SBATCH --partition=normal
#SBATCH --ntasks=1
# memory in MB
#SBATCH --mem=1024
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=results_hw4/deep_1_output_core_%04a.txt
#SBATCH --error=results_hw4/deep_1_error_core_%04a.txt
#SBATCH --time=12:00:00
#SBATCH --job-name=deep_recent
#SBATCH --mail-user=michael.montalbano@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/mcmontalbano/HW4
#SBATCH --array=0-5
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
source ~fagg/pythonenv/tensorflow/bin/activate

python hw4_base.py  -conv_size  7 3 -network 'deep' -conv_nfilters 10 20  -hidden 128 64 -dropout 0.5 -exp_index $SLURM_ARRAY_TASK_ID -epochs 1000 -L2_regularizer 0.01 -experiment_type 'basic' -dataset '/home/fagg/datasets/core50/core50_128x128' 

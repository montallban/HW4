#!/bin/bash

#SBATCH --partition=normal
#SBATCH --ntasks=1
# memory in MB
#SBATCH --mem=1024
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=results2/bmi_shoulder_l1_%04a_stdout.txt
#SBATCH --error=results2/error_bmi_shoulder_l1__%04a_.txt
#SBATCH --time=12:00:00
#SBATCH --job-name=bmi_l1_reg
#SBATCH --mail-user=michael.montalbano@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/mcmontalbano/HW3
#SBATCH --array=0-5
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
source ~fagg/pythonenv/tensorflow/bin/activate

python base.py -dropout 0.1 -exp_index $SLURM_ARRAY_TASK_ID -epochs 1000 -L2_regularizer 0.01 -experiment_type 'basic' -dataset '/home/fagg/datasets/core50/core50_128x128' 
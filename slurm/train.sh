#!/bin/bash
#SBATCH --ntasks=1                                                           # Number of tasks (see below)
#SBATCH --cpus-per-task=4                                                    # Number of CPU cores per task
#SBATCH --nodes=1                                                            # Ensure that all cores are on one machine
#SBATCH --time=3-00:00                                                       # Runtime in D-HH:MM
#SBATCH --gres=gpu:2                                                         # Request 1 GPU
#SBATCH --mem=100G                                                           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/weka/bethge/dziadzio08/opencompass/slurm/hostname_%j.out   # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/weka/bethge/dziadzio08/opencompass/slurm/hostname_%j.err    # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=END,FAIL                                                 # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sebastian.dziadzio@uni-tuebingen.de                      # Email to which notifications will be sent

# print info about current job
scontrol show job $SLURM_JOB_ID
export WANDB__SERVICE_WAIT=300
export HYDRA_FULL_ERROR=1

additional_args="$@"

source $HOME/.bashrc
source $HOME/opencompass/.venv/bin/activate


./eval.sh $additional_args
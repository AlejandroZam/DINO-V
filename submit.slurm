#!/bin/bash

#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus=1
#SBATCH --output=dino-perceiver-slurm-%J.out
#SBATCH --job-name=dino-perceiver-imnet

# Give this process 1 task (per GPU, but only one GPU), then assign eight 8per task
# (so 8 cores overall).  Then enforce that slurm assigns only CPUs that are on the
# socket closest to the GPU you get.

# If you want two GPUs:
# #SBATCH --gpus=2
# This example, however, only uses one GPU.


# Output some preliminaries before we begin
date
echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"

# Load the Python and CUDA modules
module load python/python-3.8.0-gcc-9.1.0
module load cuda/cuda-10.2

# List the modules that are loaded
module list

# Have Nvidia tell us the GPU/CPU mapping so we know
nvidia-smi topo -m

echo

# Activate the GPU version of PyTorch
source activate pytorch-1.8.0+cuda10_2

# Here we are going to run the PyTorch super_resolution example from the PyTorch examples
# GitHub Repository: https://github.com/pytorch/examples/tree/master/super_resolution

# Run PyTorch Training
echo "Training Start:"
time python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch perceiver --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --epochs 100 --data_path /datasets/ImageNet2012nonpub/train --output_dir /lustre/fs0/home/cap6412.student4/DINO/ImNet-Perceiver
echo

# You're done!
echo "Ending script..."
date
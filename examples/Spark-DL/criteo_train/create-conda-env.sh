#!/bin/bash --login

export HOROVOD_CUDA_HOME=/usr/local/cuda
export HOROVOD_NCCL_LINK=SHARED
export HOROVOD_GPU_OPERATIONS=NCCL

# Create the conda environment:
conda env create --file environment.yml --force

# Activate the conda environment:
eval "$(conda shell.bash hook)"
conda activate spark_dl

# Install tensorflow:
pip install tensorflow-gpu

# Install pytorch:
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113\
 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install horovod:
pip install horovod[tensorflow,keras,pytorch,spark]

# Check the build:
horovodrun --check-build

echo
echo "Run 'conda activate spark_dl' to activate the environment"
echo

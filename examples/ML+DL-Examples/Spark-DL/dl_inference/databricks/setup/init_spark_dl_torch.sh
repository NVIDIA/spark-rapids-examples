#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -x

# setup cuda: install cudatoolkit 11.8 via runfile approach
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit
# reset symlink and update library loading paths
rm /usr/local/cuda
ln -s /usr/local/cuda-11.8 /usr/local/cuda

# install requirements
sudo /databricks/python3/bin/pip3 install --upgrade pip

cat <<EOF > temp_requirements.txt
numpy
pandas
matplotlib
portalocker
pyarrow
pydot
scikit-learn
huggingface
datasets==3.0.1
transformers
ipykernel
ipywidgets
urllib3<2
torch
torchvision --extra-index-url https://download.pytorch.org/whl/cu118
torch-tensorrt
tensorrt --extra-index-url https://download.pytorch.org/whl/cu118
sentence_transformers
sentencepiece
nvidia-pytriton
nvidia-modelopt[all] --extra-index-url https://pypi.nvidia.com
EOF

sudo /databricks/python3/bin/pip3 install --upgrade --force-reinstall -r temp_requirements.txt

rm temp_requirements.txt

set +x

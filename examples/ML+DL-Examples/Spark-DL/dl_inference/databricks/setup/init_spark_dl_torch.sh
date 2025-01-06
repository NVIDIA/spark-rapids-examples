#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -x

# install requirements
sudo /databricks/python3/bin/pip3 install --upgrade pip

cat <<EOF > temp_requirements.txt
datasets==3.*
transformers
urllib3<2
nvidia-pytriton
torch
torchvision --extra-index-url https://download.pytorch.org/whl/cu121
torch-tensorrt
tensorrt --extra-index-url https://download.pytorch.org/whl/cu121
sentence_transformers
sentencepiece
nvidia-modelopt[all] --extra-index-url https://pypi.nvidia.com
EOF

sudo /databricks/python3/bin/pip3 install --upgrade --force-reinstall -r temp_requirements.txt
rm temp_requirements.txt

set +x

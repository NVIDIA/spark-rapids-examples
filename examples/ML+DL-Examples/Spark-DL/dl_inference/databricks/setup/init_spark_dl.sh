#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -euxo pipefail

# install requirements
sudo /databricks/python3/bin/pip3 install --upgrade pip

if [[ "${FRAMEWORK}" == "torch" ]]; then
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
elif [[ "${FRAMEWORK}" == "tf" ]]; then
    cat <<EOF > temp_requirements.txt
datasets==3.*
transformers
urllib3<2
nvidia-pytriton
EOF
else
    echo "Please export FRAMEWORK as torch or tf per README"
    exit 1
fi

sudo /databricks/python3/bin/pip3 install --upgrade --force-reinstall -r temp_requirements.txt
rm temp_requirements.txt

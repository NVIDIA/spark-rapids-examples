#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -euxo pipefail

# install requirements
sudo /databricks/python3/bin/pip3 install --upgrade pip

if [[ "${FRAMEWORK}" == "torch" ]]; then
    cat <<EOF > temp_requirements.txt
datasets==3.*
transformers
nvidia-pytriton
torch<=2.5.1
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
nvidia-pytriton
EOF
elif [[ "${FRAMEWORK}" == "vllm" ]]; then
    cat <<EOF > temp_requirements.txt
vllm==0.8.2
EOF
else
    echo "Please export FRAMEWORK as torch, tf, or vllm per README"
    exit 1
fi

sudo /databricks/python3/bin/pip3 install --upgrade --force-reinstall -r temp_requirements.txt
rm temp_requirements.txt

#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euxo pipefail

function get_metadata_attribute() {
  local -r attribute_name=$1
  local -r default_value=$2
  /usr/share/google/get_metadata_value "attributes/${attribute_name}" || echo -n "${default_value}"
}

pip install --upgrade pip

SPARK_DL_HOME=$(get_metadata_attribute spark-dl-home UNSET)
if [[ ${SPARK_DL_HOME} == "UNSET" ]]; then
    echo "Please set --metadata spark-dl-home"
    exit 1
fi

cat <<EOF > temp_requirements.txt
numpy
pandas
matplotlib
portalocker
pyarrow
pydot
scikit-learn
huggingface
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

pip install --upgrade --force-reinstall -r temp_requirements.txt
rm temp_requirements.txt

if gsutil -q stat gs://${SPARK_DL_HOME}/notebooks/**; then
    mkdir spark-dl-notebooks
    gcloud storage cp -r gs://${SPARK_DL_HOME}/notebooks/* spark-dl-notebooks/
else
    echo "The directory gs://${SPARK_DL_HOME}/notebooks/ is not accessible."
    exit 1
fi

ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ${CONDA_PREFIX}/lib/libstdc++.so.6

sudo chmod -R a+rw /home/
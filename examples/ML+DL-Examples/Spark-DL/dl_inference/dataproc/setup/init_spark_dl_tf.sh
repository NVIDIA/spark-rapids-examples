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

GCS_BUCKET=$(get_metadata_attribute gcs-bucket UNSET)
if [[ ${GCS_BUCKET} == "UNSET" ]]; then
    echo "Please set --metadata gcs-bucket"
    exit 1
fi

# mount gcs bucket as fuse
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y fuse gcsfuse
sudo mkdir -p /mnt/gcs
gcsfuse --implicit-dirs ${GCS_BUCKET} /mnt/gcs
sudo chmod -R 777 /mnt/gcs

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
tensorflow[and-cuda]
tf-keras
EOF

pip install --upgrade --force-reinstall -r temp_requirements.txt
rm temp_requirements.txt

if gsutil -q stat gs://${SPARK_DL_HOME}/notebooks/**; then
    mkdir spark-dl-notebooks
    gcloud storage cp -r gs://${SPARK_DL_HOME}/notebooks/* spark-dl-notebooks
else
    echo "The directory gs://${SPARK_DL_HOME}/notebooks/ is not accessible."
    exit 1
fi

sudo chmod -R a+rw /home/
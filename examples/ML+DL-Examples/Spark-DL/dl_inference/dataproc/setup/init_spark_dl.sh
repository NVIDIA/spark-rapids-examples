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

gcloud storage cp gs://${SPARK_DL_HOME}/requirements.txt .
pip install --upgrade --force-reinstall -r requirements.txt

gcloud storage cp gs://${SPARK_DL_HOME}/conditional_generation.ipynb notebooks/conditional_generation.ipynb

sudo chmod -R a+rw /home/
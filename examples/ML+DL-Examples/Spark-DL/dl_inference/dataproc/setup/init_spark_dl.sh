#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euxo pipefail

function get_metadata_attribute() {
  local -r attribute_name=$1
  local -r default_value=$2
  /usr/share/google/get_metadata_value "attributes/${attribute_name}" || echo -n "${default_value}"
}

RAPIDS_VERSION=$(get_metadata_attribute "rapids_version" "24.10.0")

# patch existing packages
mamba install "llvmlite<0.40,>=0.39.0dev0" "numba>=0.56.2"

pip install --upgrade pip

# # dataproc 2.1 pyarrow and arrow conda installation is not compatible with cudf
# mamba uninstall -y pyarrow arrow

# pip install cudf-cu11~=${RAPIDS_VERSION} cuml-cu11~=${RAPIDS_VERSION} cuvs-cu11~=${RAPIDS_VERSION} \
#     pylibraft-cu11~=${RAPIDS_VERSION} \
#     rmm-cu11~=${RAPIDS_VERSION} \
#     --extra-index-url=https://pypi.nvidia.com

SPARK_DL_HOME=$(get_metadata_attribute spark-dl-home UNSET)
if [[ ${SPARK_DL_HOME} == "UNSET" ]]; then
    echo "Please set --metadata spark-dl-home"
    exit 1
fi

gcloud storage cp requirements.txt gs://${SPARK_DL_HOME}/requirements.txt .
pip install --upgrade --force-reinstall -r requirements.txt

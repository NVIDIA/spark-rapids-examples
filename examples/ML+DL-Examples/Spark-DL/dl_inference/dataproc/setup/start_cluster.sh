#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -eo pipefail

# configure arguments
if [[ -z ${GCS_BUCKET} ]]; then
    echo "Please export GCS_BUCKET per README.md"
    exit 1
fi

if [[ -z ${FRAMEWORK} ]]; then
    echo "Please export FRAMEWORK as 'torch' or 'tf'"
    exit 1
fi

if [[ -z ${COMPUTE_REGION} ]]; then
    COMPUTE_REGION=$(gcloud config get-value compute/region)
    if [[ -z ${COMPUTE_REGION} ]]; then
        echo "Please export COMPUTE_REGION per README.md or set it in gcloud config."
        exit 1
    fi
fi

SPARK_DL_HOME=${SPARK_DL_HOME:-${GCS_BUCKET}/spark-dl}

# copy init script to gcs
gcloud storage cp init_spark_dl.sh gs://${SPARK_DL_HOME}/init/
INIT_PATH=gs://${SPARK_DL_HOME}/init/init_spark_dl.sh

# retrieve and upload spark-rapids initialization script to gcs
curl -LO https://raw.githubusercontent.com/GoogleCloudDataproc/initialization-actions/master/spark-rapids/spark-rapids.sh
# don't enable rapids plugin by default
sed -i '/spark.plugins=com.nvidia.spark.SQLPlugin/d' spark-rapids.sh
gcloud storage cp spark-rapids.sh gs://${SPARK_DL_HOME}/init/
# rm spark-rapids.sh

COMMON_REQUIREMENTS="numpy
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
nvidia-pytriton"

TORCH_REQUIREMENTS="${COMMON_REQUIREMENTS}
torch<=2.5.1
torchvision --extra-index-url https://download.pytorch.org/whl/cu121
torch-tensorrt
tensorrt --extra-index-url https://download.pytorch.org/whl/cu121
sentence_transformers
sentencepiece
nvidia-modelopt[all] --extra-index-url https://pypi.nvidia.com"

TF_REQUIREMENTS="${COMMON_REQUIREMENTS}
tensorflow[and-cuda]
tf-keras"

cluster_name=${USER}-spark-dl-inference-${FRAMEWORK}
if [[ ${FRAMEWORK} == "torch" ]]; then
    requirements=${TORCH_REQUIREMENTS}
    echo "========================================================="
    echo "Starting PyTorch cluster ${cluster_name}"
    echo "========================================================="
elif [[ ${FRAMEWORK} == "tf" ]]; then
    requirements=${TF_REQUIREMENTS}
    echo "========================================================="
    echo "Starting Tensorflow cluster ${cluster_name}"
    echo "========================================================="
else
    echo "Please export FRAMEWORK as torch or tf"
    exit 1
fi

if gcloud dataproc clusters list | grep -q "${cluster_name}"; then
    echo "Cluster ${cluster_name} already exists."
    exit 0
fi

CLUSTER_PARAMS=(
    --image-version=2.2-ubuntu
    --region ${COMPUTE_REGION}
    --num-workers 4
    --initialization-actions gs://${SPARK_DL_HOME}/init/spark-rapids.sh,${INIT_PATH}
    --metadata gpu-driver-provider="NVIDIA"
    --metadata gcs-bucket=${GCS_BUCKET}
    --metadata spark-dl-home=${SPARK_DL_HOME}
    --metadata requirements="${requirements}"
    --worker-local-ssd-interface=NVME
    --optional-components=JUPYTER
    --bucket ${GCS_BUCKET}
    --enable-component-gateway
    --max-idle "60m"
    --subnet=default
    --no-shielded-secure-boot
)

if [[ ${USE_L4} == "true" ]]; then
    echo "Using L4 GPUs."
    GPU_TYPE_PARAMS=(
        --master-machine-type g2-standard-8
        --worker-machine-type g2-standard-8
    )
else
    echo "Using T4 GPUs."
    GPU_TYPE_PARAMS=(
        --master-machine-type n1-standard-16
        --worker-machine-type n1-standard-16
        --worker-min-cpu-platform="Intel Skylake"
        --master-accelerator type=nvidia-tesla-t4,count=1
        --worker-accelerator type=nvidia-tesla-t4,count=1
    )
fi

gcloud dataproc clusters create ${cluster_name} "${CLUSTER_PARAMS[@]}" "${GPU_TYPE_PARAMS[@]}"

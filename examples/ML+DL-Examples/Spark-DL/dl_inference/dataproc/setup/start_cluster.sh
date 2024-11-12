#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

# configure arguments
if [[ -z ${COMPUTE_REGION} ]]; then
    echo "Please export COMPUTE_REGION per README.md"
    exit 1
fi

if [[ -z ${GCS_BUCKET} ]]; then
    echo "Please export GCS_BUCKET per README.md"
    exit 1
fi

SPARK_DL_HOME=${SPARK_DL_HOME:-${GCS_BUCKET}/spark-dl}

gpu_args=$(cat <<EOF
--master-accelerator type=nvidia-tesla-t4,count=1
--worker-accelerator type=nvidia-tesla-t4,count=1
--initialization-actions gs://${SPARK_DL_HOME}/spark-rapids.sh,gs://${SPARK_DL_HOME}/init_spark_dl.sh
--metadata gpu-driver-provider="NVIDIA"
--metadata rapids_runtime=SPARK
--metadata spark-dl-home=${SPARK_DL_HOME}
EOF
)

# start cluster if not already running
cluster_name=${CLUSTER_NAME:-"${USER}-spark-dl-gpu"}

gcloud dataproc clusters list | grep "${cluster_name}"
if [[ $? == 0 ]]; then
    echo "WARNING: Cluster ${cluster_name} is already started."
else
    set -x
    gcloud dataproc clusters create ${cluster_name} \
    --image-version=2.1-ubuntu \
    --region ${COMPUTE_REGION} \
    --master-machine-type n1-standard-16 \
    --num-workers 2 \
    --worker-min-cpu-platform="Intel Skylake" \
    --worker-machine-type n1-standard-16 \
    --num-worker-local-ssds 4 \
    --worker-local-ssd-interface=NVME \
    ${gpu_args} \
    --optional-components=JUPYTER \
    --bucket ${GCS_BUCKET} \
    --enable-component-gateway \
    --max-idle "60m" \
    --subnet=default \
    --no-shielded-secure-boot
fi
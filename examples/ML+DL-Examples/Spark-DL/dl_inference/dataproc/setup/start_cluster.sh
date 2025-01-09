#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

# configure arguments
if [[ -z ${GCS_BUCKET} ]]; then
    echo "Please export GCS_BUCKET per README.md"
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

if [[ $INIT_SRC == *init_spark_dl_torch.sh ]]; then
    init_script="gs://${SPARK_DL_HOME}/init/init_spark_dl_torch.sh"
    cluster_name=${USER}-spark-dl-inference-torch
    echo "=============================================================="
    echo "=== Starting PyTorch cluster ${cluster_name} using init_spark_dl_torch.sh ==="
    echo "=============================================================="
elif [[ $INIT_SRC == *init_spark_dl_tf.sh ]]; then
    init_script="gs://${SPARK_DL_HOME}/init/init_spark_dl_tf.sh"
    cluster_name=${USER}-spark-dl-inference-tf
    echo "=============================================================="
    echo "=== Starting TensorFlow cluster ${cluster_name} using init_spark_dl_tf.sh ==="
    echo "=============================================================="
else
    echo "Please make sure INIT_SRC is set so that the cluster knows which init script to use (_torch or _tf)."
    exit 1
fi

# retrieve and upload spark-rapids initialization script to gcs
curl -LO https://raw.githubusercontent.com/GoogleCloudDataproc/initialization-actions/master/spark-rapids/spark-rapids.sh
# don't enable plugin by default
sed -i '/spark.plugins=com.nvidia.spark.SQLPlugin/d' spark-rapids.sh
gcloud storage cp spark-rapids.sh gs://${SPARK_DL_HOME}/init/
rm spark-rapids.sh

# start cluster if not already running
gcloud dataproc clusters list | grep "${cluster_name}"
if [[ $? == 0 ]]; then
    echo "Cluster ${cluster_name} already exists. Starting it..."
    gcloud dataproc clusters start ${cluster_name} --region ${COMPUTE_REGION}
else
    gcloud dataproc clusters create ${cluster_name} \
    --image-version=2.2-ubuntu \
    --region ${COMPUTE_REGION} \
    --master-machine-type n1-standard-16 \
    --num-workers 4 \
    --worker-min-cpu-platform="Intel Skylake" \
    --worker-machine-type n1-standard-16 \
    --master-accelerator type=nvidia-tesla-t4,count=1 \
    --worker-accelerator type=nvidia-tesla-t4,count=1 \
    --initialization-actions gs://${SPARK_DL_HOME}/init/spark-rapids.sh,${init_script} \
    --metadata gpu-driver-provider="NVIDIA" \
    --metadata spark-dl-home=${SPARK_DL_HOME} \
    --worker-local-ssd-interface=NVME \
    --optional-components=JUPYTER \
    --bucket ${GCS_BUCKET} \
    --enable-component-gateway \
    --max-idle "120m" \
    --subnet=default \
    --no-shielded-secure-boot
fi

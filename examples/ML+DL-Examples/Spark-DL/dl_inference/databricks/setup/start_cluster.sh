#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -eo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <aws|azure> [tp]"
    exit 1
fi

CLOUD_PROVIDER=$1
TENSOR_PARALLEL=false

# Check if the second argument is "tp" for tensor parallelism
if [ $# -eq 2 ] && [ "$2" == "tp" ]; then
    TENSOR_PARALLEL=true
fi

if [[ "${FRAMEWORK}" != "vllm" && "${FRAMEWORK}" != "torch" && "${FRAMEWORK}" != "tf" ]]; then
    echo "Error: Please export FRAMEWORK as torch, tf, or vllm per README"
    exit 1
fi

# Modify the node types below if your Databricks account does not have these specific instance types. 
# Modify EXECUTOR_CORES=(cores per node) and EXECUTOR_GPU_AMT=(GPUs per node) accordingly.
# We recommend selecting instances with A10/L4+ GPUs for these examples.
if [[ "${CLOUD_PROVIDER}" == "aws" ]]; then
    DRIVER_NODE_TYPE="g5.2xlarge"
    
    if [[ "${TENSOR_PARALLEL}" == "true"]]; then
        # For tensor-parallelism examples, select an instance with 4 GPUs (AWS does not have 2-GPU A10/L4 instances). 
        NODE_TYPE="g5.12xlarge"
        EXECUTOR_CORES=48
        EXECUTOR_GPU_AMT=4
    else
        NODE_TYPE="g5.4xlarge"
        EXECUTOR_CORES=16
        EXECUTOR_GPU_AMT=1
    fi
elif [[ "${CLOUD_PROVIDER}" == "azure" ]]; then
    DRIVER_NODE_TYPE="Standard_NV36ads_A10_v5"
    
    if [[ "${TENSOR_PARALLEL}" == "true"]]; then
        # For tensor-parallelism examples, we need an instance with 2 GPUs.
        NODE_TYPE="Standard_NV72ads_A10_v5"
        EXECUTOR_CORES=72
        EXECUTOR_GPU_AMT=2
    else
        NODE_TYPE="Standard_NV36ads_A10_v5"
        EXECUTOR_CORES=36
        EXECUTOR_GPU_AMT=1
    fi
else
    echo "Error: Cloud provider must be either 'aws' or 'azure'"
    exit 1
fi

# Use a descriptive cluster name to indicate tensor parallelism if enabled
CLUSTER_SUFFIX="${FRAMEWORK}"
if [[ "${TENSOR_PARALLEL}" == "true" ]]; then
    CLUSTER_SUFFIX="${FRAMEWORK}-tp"
fi

# task GPU amount = executor GPU amount / executor cores
TASK_GPU_AMT=$(awk "BEGIN {print ${EXECUTOR_GPU_AMT}/${EXECUTOR_CORES}}")

json_config=$(cat <<EOF
{
    "cluster_name": "spark-dl-inference-${CLUSTER_SUFFIX}",
    "spark_version": "15.4.x-gpu-ml-scala2.12",
    "spark_conf": {
        "spark.executor.resource.gpu.amount": "${EXECUTOR_GPU_AMT}",
        "spark.python.worker.reuse": "true",
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.task.resource.gpu.amount": "${TASK_GPU_AMT}",
        "spark.executor.cores": "${EXECUTOR_CORES}"
    },
    "node_type_id": "${NODE_TYPE}",
    "driver_node_type_id": "${DRIVER_NODE_TYPE}",
    "spark_env_vars": {
        "TF_GPU_ALLOCATOR": "cuda_malloc_async",
        "FRAMEWORK": "${FRAMEWORK}"
    },
    "autotermination_minutes": 60,
    "enable_elastic_disk": true,
    "init_scripts": [
        {
            "workspace": {
                "destination": "${SPARK_DL_WS}/init_spark_dl.sh"
            }
        }
    ],
    "runtime_engine": "STANDARD",
    "num_workers": "2"
}
EOF
)

databricks clusters create --json "$json_config"

#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -eo pipefail

# Modify the node types below if your Databricks account does not have these specific instance types. 
# Modify EXECUTOR_CORES=(cores per node) accordingly.
# We recommend selecting A10/L4+ instances for these examples.
DRIVER_NODE_TYPE="Standard_NV36ads_A10_v5"

if [[ "${FRAMEWORK}" == "vllm" ]]; then
    # For vLLM tensor-parallelism examples, select an instance with **two GPUs**. 
    NODE_TYPE="Standard_NV72ads_A10_v5"
    EXECUTOR_CORES=72
    EXECUTOR_GPU_AMT=2
    TASK_GPU_AMT=$(awk "BEGIN {print ${EXECUTOR_GPU_AMT}/${EXECUTOR_CORES}}")
elif [[ "${FRAMEWORK}" == "torch" || "${FRAMEWORK}" == "tf" ]]; then
    NODE_TYPE="Standard_NV36ads_A10_v5"
    EXECUTOR_CORES=36
    EXECUTOR_GPU_AMT=1
    TASK_GPU_AMT=$(awk "BEGIN {print ${EXECUTOR_GPU_AMT}/${EXECUTOR_CORES}}")
else
    echo "Error: Please export FRAMEWORK as torch, tf, or vllm per README"
    exit 1
fi

json_config=$(cat <<EOF
{
    "cluster_name": "spark-dl-inference-${FRAMEWORK}",
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

#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -eo pipefail

# configure arguments
if [[ -z ${INIT_DEST} ]]; then
    echo "Please make sure INIT_DEST is exported per README.md"
    exit 1
fi

if [[ -z ${FRAMEWORK} ]]; then
    echo "Please make sure FRAMEWORK is exported to torch or tf per README.md"
    exit 1
fi

# For LLM examples: modify the node_type_id and driver_node_type_id below to use A10 GPUs (e.g., Standard_NV6ads_A10_v5)
# and modify task.resource.gpu.amount and executor.cores accordingly (e.g., 0.16667 and 6 respectively)
json_config=$(cat <<EOF
{
    "cluster_name": "spark-dl-inference-${FRAMEWORK}",
    "spark_version": "15.4.x-gpu-ml-scala2.12",
    "spark_conf": {
        "spark.executor.resource.gpu.amount": "1",
        "spark.python.worker.reuse": "true",
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.task.resource.gpu.amount": "0.125",
        "spark.executor.cores": "8"
    },
    "node_type_id": "Standard_NC8as_T4_v3",
    "driver_node_type_id": "Standard_NC8as_T4_v3",
    "spark_env_vars": {
        "TF_GPU_ALLOCATOR": "cuda_malloc_async",
        "FRAMEWORK": "${FRAMEWORK}"
    },
    "autotermination_minutes": 60,
    "enable_elastic_disk": true,
    "init_scripts": [
        {
            "workspace": {
                "destination": "${INIT_DEST}"
            }
        }
    ],
    "runtime_engine": "STANDARD",
    "num_workers": 4
}
EOF
)

databricks clusters create --json "$json_config"

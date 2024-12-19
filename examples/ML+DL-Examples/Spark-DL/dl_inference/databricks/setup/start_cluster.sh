#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

# configure arguments
if [[ -z ${INIT_DEST} ]]; then
    echo "Please make sure INIT_DEST is exported per README.md"
    exit 1
fi

if [[ -z ${CLUSTER_NAME} ]]; then
    echo "Please make sure CLUSTER_NAME is exported per README.md"
    exit 1
fi

json_config=$(cat <<EOF
{
    "cluster_name": "${CLUSTER_NAME}",
    "spark_version": "13.3.x-gpu-ml-scala2.12",
    "spark_conf": {
        "spark.executor.resource.gpu.amount": "1",
        "spark.python.worker.reuse": "true",
        "spark.task.resource.gpu.amount": "0.125",
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.executor.cores": "8"
    },
    "node_type_id": "Standard_NC8as_T4_v3",
    "driver_node_type_id": "Standard_NC8as_T4_v3",
    "spark_env_vars": {
        "TF_GPU_ALLOCATOR": "cuda_malloc_async"
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
#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

# configure arguments
if [[ -z ${INIT_PATH} ]]; then
    echo "Please export INIT_PATH per README.md"
    exit 1
fi

json_config=$(cat <<EOF
{
    "cluster_name": "spark-dl-inference",
    "spark_version": "13.3.x-gpu-ml-scala2.12",
    "spark_conf": {
        "spark.task.resource.gpu.amount": "0.125",
        "spark.executor.cores": "8",
        "spark.executor.resource.gpu.amount": "1",
        "spark.sql.execution.arrow.pyspark.enabled": "true",
    },
    "node_type_id": "Standard_NC8as_T4_v3",
    "driver_node_type_id": "Standard_NC8as_T4_v3",
    "autotermination_minutes": 60,
    "enable_elastic_disk": true,
    "init_scripts": [
        {
            "workspace": {
                "destination": "${INIT_PATH}"
            }
        }
    ],
    "runtime_engine": "STANDARD",
    "num_workers": 2
}
EOF
)

databricks clusters create --json "$json_config"
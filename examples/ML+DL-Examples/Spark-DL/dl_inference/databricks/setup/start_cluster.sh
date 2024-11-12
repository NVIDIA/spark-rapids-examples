#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

# configure arguments
if [[ -z ${INIT_PATH} ]]; then
    echo "Please export INIT_PATH per README.md"
    exit 1
fi

databricks clusters create --json <<EOF
{
    "cluster_name": "spark-dl-inference",
    "spark_version": "13.3.x-gpu-ml-scala2.12",
    "spark_conf": {
        "spark.task.resource.gpu.amount": "0.1666",
        "spark.python.worker.reuse": "true",
        "spark.executorEnv.PYTHONPATH": "/databricks/jars/rapids-4-spark_2.12-24.10.0.jar:/databricks/spark/python:/databricks/python3",
        "spark.sql.pyspark.jvmStacktrace.enabled": "true",
        "spark.sql.execution.arrow.maxRecordsPerBatch": "10000",
        "spark.executor.cores": "6",
        "spark.rapids.memory.gpu.minAllocFraction": "0.0001",
        "spark.sql.execution.pyspark.udf.simplifiedTraceback.enabled": "false",
        "spark.plugins": "com.nvidia.spark.SQLPlugin",
        "spark.locality.wait": "0s",
        "spark.sql.cache.serializer": "com.nvidia.spark.ParquetCachedBatchSerializer",
        "spark.rapids.memory.gpu.pooling.enabled": "false",
        "spark.executor.resource.gpu.amount": "1",
        "spark.rapids.sql.explain": "NONE",
        "spark.sql.execution.sortBeforeRepartition": "false",
        "spark.rapids.sql.python.gpu.enabled": "true",
        "spark.rapids.memory.pinnedPool.size": "2G",
        "spark.task.maxFailures": "1",
        "spark.python.daemon.module": "rapids.daemon_databricks",
        "spark.rapids.ml.uvm.enabled": "true",
        "spark.rapids.sql.batchSizeBytes": "512m",
        "spark.sql.adaptive.enabled": "false",
        "spark.rapids.sql.format.parquet.reader.type": "MULTITHREADED",
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.rapids.sql.format.parquet.multiThreadedRead.maxNumFilesParallel": "20",
        "spark.sql.files.maxPartitionBytes": "512m",
        "spark.rapids.sql.multiThreadedRead.numThreads": "20",
        "spark.rapids.sql.concurrentGpuTasks": "4"
    },
    "node_type_id": "Standard_NC6s_v3",
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
    "num_workers": 8
}
EOF
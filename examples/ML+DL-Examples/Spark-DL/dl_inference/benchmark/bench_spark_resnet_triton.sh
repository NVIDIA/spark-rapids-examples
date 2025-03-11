#! /bin/bash

spark-submit \
  --master spark://$(hostname):7077 \
  --num-executors 1 \
  --executor-cores 16 \
  --executor-memory 32g \
  --conf spark.executor.resource.gpu.amount=1 \
  --conf spark.task.resource.gpu.amount=0.0625 \
  --conf spark.task.maxFailures=1 \
  --conf spark.sql.execution.arrow.pyspark.enabled=true \
  --conf spark.python.worker.reuse=true \
  --conf spark.pyspark.python=${CONDA_PREFIX}/bin/python \
  --conf spark.pyspark.driver.python=${CONDA_PREFIX}/bin/python \
  --conf spark.locality.wait=0s \
  --conf spark.sql.adaptive.enabled=false \
  --conf spark.sql.execution.sortBeforeRepartition=false \
  --conf spark.sql.files.minPartitionNum=16 \
  bench_spark_resnet_triton.py

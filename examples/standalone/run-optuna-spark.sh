# Copyright (c) 2024, NVIDIA CORPORATION.

export PYSPARK_PYTHON=./environment/bin/python

$SPARK_HOME/bin/spark-submit \
  --master spark://$(hostname):7077 \
  --conf spark.executor.cores=8 \
  --conf spark.executor.memory=16g \
  --archives ../optuna-env.tar.gz#environment \
  ../optuna-mysql-spark.py \
    --tasks 2 \
    --jobs 2 \

# Copyright (c) 2024, NVIDIA CORPORATION.

$SPARK_HOME/bin/spark-submit \
  --master spark://$(hostname):7077 \
  --conf spark.executor.cores=8 \
  --conf spark.executor.memory=16g \
  --conf spark.executorEnv.PYSPARK_PYTHON=./environment/bin/python \
  --archives ../optuna-env.tar.gz#environment \
  ../joblibspark-simple.py \
    --tasks 2 \
    --jobs 2 \
    --localhost \

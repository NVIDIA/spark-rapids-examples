# Copyright (c) 2024, NVIDIA CORPORATION.

$SPARK_HOME/bin/spark-submit \
  --master spark://$(hostname):7077 \
  --conf spark.executor.cores=8 \
  --conf spark.executor.resource.gpu.amount=1 \
  --conf spark.task.resource.gpu.amount=1 \
  --conf spark.executor.memory=16g \
  --conf spark.executorEnv.PYSPARK_PYTHON=./environment/bin/python \
  --archives ../optuna-env.tar.gz#environment \
  ../joblibspark-xgboost.py \
    --filepath $FILEPATH \
    --tasks 2 \
    --jobs 2 \
    --localhost \

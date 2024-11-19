# Copyright (c) 2024, NVIDIA CORPORATION.

if [[ -z $SPARK_HOME ]]; then
  echo "Please export SPARK_HOME to the Spark path"
  exit 1
fi

if [[ -z $FILEPATH ]]; then
  echo "Please export FILEPATH to the path of the dataset"
  exit 1
fi

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

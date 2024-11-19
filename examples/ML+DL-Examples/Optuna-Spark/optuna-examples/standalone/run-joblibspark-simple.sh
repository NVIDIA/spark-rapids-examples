# Copyright (c) 2024, NVIDIA CORPORATION.

if [[ -z $SPARK_HOME ]]; then
  echo "Please export SPARK_HOME to the Spark path"
  exit 1
fi

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

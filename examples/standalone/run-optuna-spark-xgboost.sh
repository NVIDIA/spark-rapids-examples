export PYSPARK_PYTHON=./environment/bin/python

$SPARK_HOME/bin/spark-submit \
  --master spark://$(hostname):7077 \
  --conf spark.executor.cores=8 \
  --conf spark.executor.resource.gpu.amount=1 \
  --conf spark.task.resource.gpu.amount=1 \
  --conf spark.executor.memory=16g \
  --archives ../optuna-env.tar.gz#environment \
  ../optuna-mysql-xgboost-spark.py \
    --filepath /path/to/winequality-red.csv \
    --tasks 2 \
    --jobs 2 \

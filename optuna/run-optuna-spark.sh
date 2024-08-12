export PYSPARK_DRIVER_PYTHON=/home/bobwang/anaconda3/envs/optuna-spark/bin/python # Do not set in cluster modes.
export PYSPARK_PYTHON=./environment/bin/python

spark-submit \
  --master spark://10.19.129.248:7077 \
  --conf spark.executor.cores=1 \
  --conf spark.executor.memory=60g \
  --conf spark.task.cpus=1 \
  --archives optuna-env.tar.gz#environment \
    optuna-mysql-spark.py

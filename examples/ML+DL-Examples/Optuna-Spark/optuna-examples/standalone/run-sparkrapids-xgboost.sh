# Copyright (c) 2024, NVIDIA CORPORATION.

if [[ -z ${SCRIPT} ]]; then
    echo "Please select the implementation with: export SCRIPT=sparkrapids-xgboost-<implementation>.py"
    exit 1
fi

$SPARK_HOME/bin/spark-submit \
  --master spark://$(hostname):7077 \
  --conf spark.executor.cores=8 \
  --conf spark.executor.resource.gpu.amount=1 \
  --conf spark.task.resource.gpu.amount=1 \
  --conf spark.executor.memory=16g \
  --conf spark.task.maxFailures=1 \
  --conf spark.rpc.message.maxSize=1024 \
  --conf spark.sql.pyspark.jvmStacktrace.enabled=true \
  --conf spark.sql.execution.pyspark.udf.simplifiedTraceback.enabled=false \
  --conf spark.sql.execution.arrow.pyspark.enabled=true \
  --conf spark.python.worker.reuse=true \
  --conf spark.rapids.ml.uvm.enabled=true \
  --conf spark.jars=$RAPIDS_JAR \
  --conf spark.executorEnv.PYSPARK_PYTHON=./environment/bin/python \
  --conf spark.rapids.memory.gpu.minAllocFraction=0.0001 \
  --conf spark.plugins=com.nvidia.spark.SQLPlugin \
  --conf spark.locality.wait=0s \
  --conf spark.sql.cache.serializer=com.nvidia.spark.ParquetCachedBatchSerializer \
  --conf spark.rapids.memory.gpu.pooling.enabled=false \
  --conf spark.sql.execution.sortBeforeRepartition=false \
  --conf spark.rapids.sql.format.parquet.reader.type=MULTITHREADED \
  --conf spark.rapids.sql.format.parquet.multiThreadedRead.maxNumFilesParallel=20 \
  --conf spark.rapids.sql.multiThreadedRead.numThreads=20 \
  --conf spark.rapids.sql.python.gpu.enabled=true \
  --conf spark.rapids.memory.pinnedPool.size=2G \
  --conf spark.rapids.sql.batchSizeBytes=512m \
  --conf spark.sql.adaptive.enabled=false \
  --conf spark.sql.files.maxPartitionBytes=512m \
  --conf spark.rapids.sql.concurrentGpuTasks=4 \
  --conf spark.sql.execution.arrow.maxRecordsPerBatch=20000 \
  --conf spark.rapids.sql.explain=NONE \
  --archives ../optuna-env.tar.gz#environment \
  ../${SCRIPT} \
    --filepath $FILEPATH \
    --tasks 4 \
    --localhost \

#!/bin/sh

export SPARK_HOME=/spark-3.1.2-bin-hadoop3.2
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --allow-root"
$SPARK_HOME/bin/pyspark --master spark://127.0.0.1:7077            \
--conf spark.plugins=com.nvidia.spark.SQLPlugin        \
--driver-memory 10G                                    \
--executor-memory 40G                                  \
--conf spark.driver.extraJavaOptions="-Duser.timezone=UTC" \
--conf spark.executor.extraJavaOptions="-Duser.timezone=UTC -Dai.rapids.cudf.prefer-pinned=true" \
--conf spark.driver.extraJavaOptions="-Duser.timezone=UTC" \
--conf spark.rapids.sql.expression.UnixTimestamp=true \
--conf spark.executor.resource.gpu.amount=1 \
--conf spark.rapids.sql.explain=NOT_ON_GPU \
--conf spark.rapids.cudfVersionOverride=true \
--conf spark.rapids.memory.gpu.pool=NONE \
--conf spark.task.resource.gpu.amount=0.01 \
--conf spark.rapids.sql.concurrentGpuTasks=2 \
--conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
--files $SPARK_HOME/examples/src/main/scripts/getGpusResources.sh


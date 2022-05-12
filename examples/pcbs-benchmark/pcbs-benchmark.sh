BENCH_JAR="/home/rjafri/spark-rapids-examples/examples/pcbs-benchmark/target/pcbs-benchmark-1.0-SNAPSHOT.jar"
PLUGIN_JAR="lib/rapids-4-spark_2.12-22.06.0-20220512.134355-33-cuda11.jar"
SPARK_HOME="/home/rjafri/spark-3.1.1-bin-hadoop3.2"

DRIVE_MEMORY=32 # GB
PINNED_MEMORY=16  # GB
EXECUTOR_MEMORY=32 # GB
EXECUTOR_CORES=16 
NUM_EXECUTOR=4

$SPARK_HOME/bin/spark-submit --master spark://raza-linux-2:7077 \
--driver-memory ${DRIVE_MEMORY}G \
--executor-memory ${EXECUTOR_MEMORY}G \
--num-executors $NUM_EXECUTOR \
--conf spark.rapids.sql.enabled=true \
--conf spark.task.cpus=1 \
--conf spark.driver.maxResultSize=0 \
--conf spark.locality.wait=0 \
--conf spark.executor.heartbeatInterval=20s \
--conf spark.sql.shuffle.partitions=200 \
--conf spark.sql.files.maxPartitionBytes=1024m \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.plugins=com.nvidia.spark.SQLPlugin \
--conf spark.rapids.sql.incompatibleOps.enabled=true \
--conf spark.rapids.sql.variableFloatAgg.enabled=true \
--conf spark.rapids.sql.concurrentGpuTasks=4 \
--conf spark.sql.inMemoryColumnarStorage.compressed=true \
--conf spark.sql.adaptive.enabled=true \
--conf spark.rapids.memory.gpu.pool=ARENA \
--conf spark.rapids.memory.pinnedPool.size=${PINNED_MEMORY}G \
--conf spark.rapids.memory.gpu.allocFraction=0.9 \
--conf spark.rapids.sql.castStringToInteger.enabled=true \
--conf spark.driver.extraJavaOptions='-ea -Duser.timezone=UTC' \
--conf spark.executor.extraJavaOptions='-ea -Duser.timezone=UTC -Dai.rapids.cudf.prefer-pinned=true' \
--conf spark.task.resource.gpu.amount=0.08 \
--conf spark.executor.resource.gpu.amount=1 \
--conf spark.rapids.shuffle.ucx.enabled=false \
--conf spark.rapids.sql.explain=ALL \
--conf spark.rapids.sql.decimalType.enabled=true \
--conf spark.sql.parquet.filterPushdown=true \
--conf spark.sql.cache.serializer=com.nvidia.spark.ParquetCachedBatchSerializer \
--conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
--files ${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh \
--class com.nvidia.BenchmarkPcbs \
--jars ${PLUGIN_JAR} \
${BENCH_JAR} $1

$SPARK_HOME/bin/spark-submit --master spark://raza-linux-2:7077 \
--driver-memory ${DRIVE_MEMORY}G \
--executor-memory ${EXECUTOR_MEMORY}G \
--num-executors $NUM_EXECUTOR \
--conf spark.rapids.sql.enabled=true \
--conf spark.task.cpus=1 \
--conf spark.driver.maxResultSize=0 \
--conf spark.locality.wait=0 \
--conf spark.executor.heartbeatInterval=20s \
--conf spark.sql.shuffle.partitions=200 \
--conf spark.sql.files.maxPartitionBytes=1024m \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.plugins=com.nvidia.spark.SQLPlugin \
--conf spark.rapids.sql.incompatibleOps.enabled=true \
--conf spark.rapids.sql.variableFloatAgg.enabled=true \
--conf spark.rapids.sql.concurrentGpuTasks=4 \
--conf spark.sql.inMemoryColumnarStorage.compressed=true \
--conf spark.sql.adaptive.enabled=true \
--conf spark.rapids.memory.gpu.pool=ARENA \
--conf spark.rapids.memory.pinnedPool.size=${PINNED_MEMORY}G \
--conf spark.rapids.memory.gpu.allocFraction=0.9 \
--conf spark.rapids.sql.castStringToInteger.enabled=true \
--conf spark.driver.extraJavaOptions='-ea -Duser.timezone=UTC' \
--conf spark.executor.extraJavaOptions='-ea -Duser.timezone=UTC -Dai.rapids.cudf.prefer-pinned=true' \
--conf spark.task.resource.gpu.amount=0.08 \
--conf spark.executor.resource.gpu.amount=1 \
--conf spark.rapids.shuffle.ucx.enabled=false \
--conf spark.rapids.sql.explain=ALL \
--conf spark.rapids.sql.decimalType.enabled=true \
--conf spark.sql.parquet.filterPushdown=true \
--conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
--files ${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh \
--class com.nvidia.BenchmarkDefa \
--jars ${PLUGIN_JAR} \
${BENCH_JAR} $1

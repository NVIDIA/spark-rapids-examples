BENCH_JAR="/home/rjafri/spark-rapids-examples/examples/pcbs-benchmark/target/pcbs-benchmark-1.0-SNAPSHOT.jar"
PLUGIN_JAR="lib/rapids-4-spark_2.12-22.06.0-20220512.134355-33-cuda11.jar"
SPARK_HOME="/home/rjafri/spark-3.1.1-bin-hadoop3.2"

DRIVE_MEMORY=32 # GB
SPARK_RAPIDS_MEMORY_PINNEDPOOL_SIZE=16 #GB
EXECUTOR_MEMORY=32 # GB
EXECUTOR_CORES=16 
NUM_EXECUTOR=4

SPARK_TASK_CPUS=1
SPARK_DRIVER_MAXRESULTSIZE=0
SPARK_LOCALITY_WAIT=0
SPARK_EXECUTOR_HEARTBEATINTERVAL=20
SPARK_SQL_SHUFFLE.PARTITIONS=200
SPARK_SQL_FILES_MAXPARTITIONBYTES=1024
SPARK_DYNAMICALLOCATION_ENABLED=false
SPARK_PLUGINS="com.nvidia.spark.SQLPlugin"
SPARK_RAPIDS_SQL_INCOMPATIBLEOPS_ENABLED="true"
SPARK_RAPIDS_SQL_VARIABLEFLOATAGG_ENABLED="true"
SPARK_RAPIDS_SQL_CONCURRENTGPUTASKS=4
SPARK_SQL_INMEMORYCOLUMNARSTORAGE_COMPRESSED="true"
SPARK_SQL_ADAPTIVE_ENABLED="true"
SPARK_RAPIDS_MEMORY_GPU_POOL="ARENA"
SPARK_RAPIDS_MEMORY_GPU_ALLOCFRACTION=0.9
SPARK_RAPIDS_SQL_CASTSTRINGTOINTEGER_ENABLED="true"
SPARK_DRIVER_EXTRAJAVAOPTIONS="-ea -Duser.timezone=UTC"
SPARK_EXECUTOR_EXTRAJAVAOPTIONS="-ea -Duser.timezone=UTC -Dai.rapids.cudf.prefer-pinned=true"
SPARK_TASK_RESOURCE_GPU_AMOUNT=0.8
SPARK_EXECUTOR_RESOURCE_GPU_AMOUNT=1
SPARK_RAPIDS_SHUFFLE_UCX_ENABLED="false"
SPARK_RAPIDS_SQL_DECIMALTYPE_ENABLED="true" 
SPARK_SQL_PARQUET_FILTERPUSHDOWN="true"
SPARK_EXECUTOR_RESOURCE_GPU_DISCOVERYSCRIPT="./getGpusResources.sh"
FILES="${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh"

# Run with PCBS 
$SPARK_HOME/bin/spark-submit --master spark://raza-linux-2:7077 \
--driver-memory ${DRIVE_MEMORY}G \
--executor-memory ${EXECUTOR_MEMORY}G \
--num-executors $NUM_EXECUTOR \
--conf spark.task.cpus=${SPARK_TASK_CPUS} \
--conf spark.driver.maxResultSize=${SPARK_DRIVER_MAXRESULTSIZE} \
--conf spark.locality.wait=${SPARK_LOCALITY_WAIT} \
--conf spark.executor.heartbeatInterval=${SPARK_EXECUTOR_HEARTBEATINTERVAL}s \
--conf spark.sql.shuffle.partitions=${SPARK_SQL_SHUFFLE_PARTITIONS} \
--conf spark.sql.files.maxPartitionBytes=${SPARK_SQL_FILES_MAXPARTITIONBYTES}m \
--conf spark.dynamicAllocation.enabled=${SPARK_DYNAMICALLOCATION_ENABLED} \
--conf spark.plugins=${SPARK_PLUGINS} \
--conf spark.rapids.sql.incompatibleOps.enabled=${SPARK_RAPIDS_SQL_INCOMPATIBLEOPS_ENABLED} \
--conf spark.rapids.sql.variableFloatAgg.enabled=${SPARK_RAPIDS_SQL_VARIABLEFLOATAGG_ENABLED} \
--conf spark.rapids.sql.concurrentGpuTasks=${CONCURRENT_GPU_TASKS} \
--conf spark.sql.inMemoryColumnarStorage.compressed=${SPARK_SQL_INMEMORYCOLUMNARSTORAGE_COMPRESSED} \
--conf spark.sql.adaptive.enabled=${SPARK_SQL_ADAPTIVE_ENABLED} \
--conf spark.rapids.memory.gpu.pool=${SPARK_RAPIDS_MEMORY_GPU_POOL} \
--conf spark.rapids.memory.pinnedPool.size=${SPARK_RAPIDS_MEMORY_PINNEDPOOL_SIZE}G \
--conf spark.rapids.memory.gpu.allocFraction=${SPARK_RAPIDS_MEMORY_GPU_ALLOCFRACTION} \
--conf spark.rapids.sql.castStringToInteger.enabled=${SPARK_RAPIDS_SQL_CASTSTRINGTOINTEGER_ENABLED} \
--conf spark.driver.extraJavaOptions=${SPARK_DRIVER_EXTRAJAVAOPTIONS} \
--conf spark.executor.extraJavaOptions=${SPARK_EXECUTOR_EXTRAJAVAOPTIONS} \
--conf spark.task.resource.gpu.amount=${SPARK_TASK_RESOURCE_GPU_AMOUNT} \
--conf spark.executor.resource.gpu.amount=${SPARK_EXECUTOR_RESOURCE_GPU_AMOUNT} \
--conf spark.rapids.shuffle.ucx.enabled=${SPARK_RAPIDS_SHUFFLE_UCX_ENABLED} \
--conf spark.rapids.sql.decimalType.enabled=${SPARK_RAPIDS_SQL_DECIMALTYPE_ENABLED} \
--conf spark.sql.parquet.filterPushdown=${SPARK_SQL_PARQUET_FILTERPUSHDOWN} \
--conf spark.executor.resource.gpu.discoveryScript=${SPARK_EXECUTOR_RESOURCE_GPU_DISCOVERYSCRIPT} \
--files ${FILES} \
--conf spark.sql.cache.serializer=com.nvidia.spark.ParquetCachedBatchSerializer \
--class com.nvidia.BenchmarkPcbs \
--jars ${PLUGIN_JAR} \
${BENCH_JAR} $1

# Run with DefaultCachedBatchSerializer
$SPARK_HOME/bin/spark-submit --master spark://raza-linux-2:7077 \
--driver-memory ${DRIVE_MEMORY}G \
--executor-memory ${EXECUTOR_MEMORY}G \
--num-executors $NUM_EXECUTOR \
--conf spark.task.cpus=${SPARK_TASK_CPUS} \
--conf spark.driver.maxResultSize=${SPARK_DRIVER_MAXRESULTSIZE} \
--conf spark.locality.wait=${SPARK_LOCALITY_WAIT} \
--conf spark.executor.heartbeatInterval=${SPARK_EXECUTOR_HEARTBEATINTERVAL}s \
--conf spark.sql.shuffle.partitions=${SPARK_SQL_SHUFFLE_PARTITIONS} \
--conf spark.sql.files.maxPartitionBytes=${SPARK_SQL_FILES_MAXPARTITIONBYTES}m \
--conf spark.dynamicAllocation.enabled=${SPARK_DYNAMICALLOCATION_ENABLED} \
--conf spark.plugins=${SPARK_PLUGINS} \
--conf spark.rapids.sql.incompatibleOps.enabled=${SPARK_RAPIDS_SQL_INCOMPATIBLEOPS_ENABLED} \
--conf spark.rapids.sql.variableFloatAgg.enabled=${SPARK_RAPIDS_SQL_VARIABLEFLOATAGG_ENABLED} \
--conf spark.rapids.sql.concurrentGpuTasks=${CONCURRENT_GPU_TASKS} \
--conf spark.sql.inMemoryColumnarStorage.compressed=${SPARK_SQL_INMEMORYCOLUMNARSTORAGE_COMPRESSED} \
--conf spark.sql.adaptive.enabled=${SPARK_SQL_ADAPTIVE_ENABLED} \
--conf spark.rapids.memory.gpu.pool=${SPARK_RAPIDS_MEMORY_GPU_POOL} \
--conf spark.rapids.memory.pinnedPool.size=${SPARK_RAPIDS_MEMORY_PINNEDPOOL_SIZE}G \
--conf spark.rapids.memory.gpu.allocFraction=${SPARK_RAPIDS_MEMORY_GPU_ALLOCFRACTION} \
--conf spark.rapids.sql.castStringToInteger.enabled=${SPARK_RAPIDS_SQL_CASTSTRINGTOINTEGER_ENABLED} \
--conf spark.driver.extraJavaOptions=${SPARK_DRIVER_EXTRAJAVAOPTIONS} \
--conf spark.executor.extraJavaOptions=${SPARK_EXECUTOR_EXTRAJAVAOPTIONS} \
--conf spark.task.resource.gpu.amount=${SPARK_TASK_RESOURCE_GPU_AMOUNT} \
--conf spark.executor.resource.gpu.amount=${SPARK_EXECUTOR_RESOURCE_GPU_AMOUNT} \
--conf spark.rapids.shuffle.ucx.enabled=${SPARK_RAPIDS_SHUFFLE_UCX_ENABLED} \
--conf spark.rapids.sql.decimalType.enabled=${SPARK_RAPIDS_SQL_DECIMALTYPE_ENABLED} \
--conf spark.sql.parquet.filterPushdown=${SPARK_SQL_PARQUET_FILTERPUSHDOWN} \
--conf spark.executor.resource.gpu.discoveryScript=${SPARK_EXECUTOR_RESOURCE_GPU_DISCOVERYSCRIPT} \
--files ${FILES} \
--class com.nvidia.BenchmarkDefa \
--jars ${PLUGIN_JAR} \
${BENCH_JAR} $1

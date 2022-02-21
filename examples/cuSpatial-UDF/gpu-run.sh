#!/bin/bash

CASE_PATH=taxi_zones
SHAPE_FILE_NAME=$CASE_PATH
SHAPE_FILE_DIR=/data/cuspatial/$CASE_PATH/shape/
DATA_IN_PATH=/data/cuspatial/$CASE_PATH/pt/
DATA_OUT_PATH=/data/cuspatial/out/gpu/

rm -rf $DATA_OUT_PATH
JARS=$HOME/work/files/jars

JARS_PATH=$JARS/cudf.jar,$JARS/rapids.jar,./target/cuspatial-udf-22.02.jar
JARS_CLASS_PATH=$JARS/cudf.jar:$JARS/rapids.jar:./target/cuspatial-udf-22.02.jar

#spark-submit \
spark-submit --master spark://$HOSTNAME:7077 \
--name "Gpu Spatial Join UDF" \
--executor-memory 50G \
--executor-cores 10 \
--conf spark.task.cpus=1 \
--conf spark.default.parallelism=3 \
--conf spark.sql.adaptive.enabled=false \
--conf spark.sql.shuffle.partitions=100 \
--conf spark.sql.files.maxPartitionBytes=2GB \
--conf spark.driver.extraJavaOptions="-Duser.timezone=UTC" \
--conf spark.executor.extraJavaOptions="-Duser.timezone=UTC" \
--conf spark.plugins=com.nvidia.spark.SQLPlugin \
--conf spark.rapids.memory.pinnedPool.size=16G \
--conf spark.rapids.sql.batchSizeBytes=1GB \
--conf spark.rapids.sql.concurrentGpuTasks=1 \
--conf spark.rapids.sql.enabled=true \
--conf spark.rapids.sql.explain=all \
--conf spark.rapids.sql.format.orc.write.enabled=true \
--conf spark.executor.resource.gpu.amount=1 \
--conf spark.executor.extraClassPath=$JARS_CLASS_PATH \
--conf spark.driver.extraClassPath=$JARS_CLASS_PATH \
--jars $JARS_PATH \
--files $SHAPE_FILE_DIR/$SHAPE_FILE_NAME.shp,$SHAPE_FILE_DIR/$SHAPE_FILE_NAME.shx \
spatial_join.py $DATA_IN_PATH $DATA_OUT_PATH

#--conf spark.rapids.sql.metrics.level=DEBUG \
#--conf spark.rapids.memory.gpu.pool=NONE \

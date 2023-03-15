#!/bin/bash
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

ML_JAR=/root/.m2/repository/com/nvidia/rapids-4-spark-ml_2.12/23.04.0-SNAPSHOT/rapids-4-spark-ml_2.12-23.04.0-SNAPSHOT.jar
PLUGIN_JAR=/root/.m2/repository/com/nvidia/rapids-4-spark_2.12/23.04.0-SNAPSHOT/rapids-4-spark_2.12-23.04.0-SNAPSHOT.jar

$SPARK_HOME/bin/spark-submit \
--master spark://127.0.0.1:7077  \
--conf spark.executor.cores=12         \
--conf spark.executor.instances=2      \
--driver-memory 30G          \
--executor-memory 30G          \
--conf spark.driver.maxResultSize=8G          \
--conf spark.rapids.sql.enabled=true \
--conf spark.plugins=com.nvidia.spark.SQLPlugin \
--conf spark.rapids.memory.gpu.allocFraction=0.35 \
--conf spark.rapids.memory.gpu.maxAllocFraction=0.6 \
--conf spark.task.resource.gpu.amount=0.08 \
--conf spark.executor.extraClassPath=$ML_JAR:$PLUGIN_JAR \
--conf spark.driver.extraClassPath=$ML_JAR:$PLUGIN_JAR \
--conf spark.executor.resource.gpu.amount=1  \
--conf spark.rpc.message.maxSize=2046 \
--conf spark.executor.heartbeatInterval=500s \
--conf spark.network.timeout=1000s \
--jars $ML_JAR,$PLUGIN_JAR \
--class com.nvidia.spark.examples.pca.Main \
/workspace/target/PCAExample-23.04.0-SNAPSHOT.jar

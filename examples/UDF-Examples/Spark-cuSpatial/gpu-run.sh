#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

# change to your spark folder
SPARK_HOME=${SPARK_HOME:-/data/spark-3.2.0-bin-hadoop3.2}

# change this path to your root path for the dataset
ROOT_PATH=${ROOT_PATH:-/data/cuspatial_data}
# Extract the sample dataset in ../../datasets/cuspatial_data.tar.gz
# Copy the polygons and points data into the root path or change the root path to where they are
SHAPE_FILE_DIR=$ROOT_PATH/polygons
SHAPE_FILE_NAME="polygons"
DATA_IN_PATH=$ROOT_PATH/points
DATA_OUT_PATH=$ROOT_PATH/output

rm -rf $DATA_OUT_PATH

# the path to keep the jars of spark-rapids & spark-cuspatial
JARS=$ROOT_PATH/jars

JARS_PATH=${JARS_PATH:-$JARS/rapids-4-spark_2.12-23.04.0-SNAPSHOT.jar,$JARS/spark-cuspatial-23.04.0-SNAPSHOT.jar}

$SPARK_HOME/bin/spark-submit --master spark://$HOSTNAME:7077 \
--name "Gpu Spatial Join UDF" \
--executor-memory 20G \
--executor-cores 10 \
--conf spark.task.cpus=1 \
--conf spark.sql.adaptive.enabled=false \
--conf spark.plugins=com.nvidia.spark.SQLPlugin \
--conf spark.rapids.sql.explain=all \
--conf spark.executor.resource.gpu.amount=1 \
--conf spark.cuspatial.sql.udf.shapeFileName="$SHAPE_FILE_NAME.shp" \
--conf spark.driver.extraLibraryPath=YOUR_LD_LIBRARY_PATH \
--conf spark.executor.extraLibraryPath=YOUR_LD_LIBRARY_PATH \
--jars $JARS_PATH \
--files $SHAPE_FILE_DIR/$SHAPE_FILE_NAME.shp,$SHAPE_FILE_DIR/$SHAPE_FILE_NAME.shx \
spatial_join.py $DATA_IN_PATH $DATA_OUT_PATH

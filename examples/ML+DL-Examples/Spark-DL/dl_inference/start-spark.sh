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

# Check if SPARK_HOME is set
if [ -z "$SPARK_HOME" ]; then
  echo "Please set the SPARK_HOME environment variable before running this script."
  exit 1
fi

# Start Jupyter with PySpark
echo "Launching PySpark with Jupyter..."
${SPARK_HOME}/bin/pyspark --master ${MASTER} \
--driver-memory 10G \
--executor-memory 8G \
--conf spark.task.maxFailures=1 \
--conf spark.locality.wait=0s \
--conf spark.sql.execution.arrow.pyspark.enabled=true \
--conf spark.sql.execution.arrow.maxRecordsPerBatch=1000 \
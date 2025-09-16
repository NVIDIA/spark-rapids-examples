#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
set -ex

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd "$SCRIPTPATH"

if [[ $( echo ${SKIP_TESTS} | tr '[:upper:]' '[:lower:]' ) == "true" ]];
then
    echo "PYTHON INTEGRATION TESTS SKIPPED..."
    exit 0
elif [[ -z "$SPARK_HOME" ]];
then
    >&2 echo "SPARK_HOME IS NOT SET CANNOT RUN PYTHON INTEGRATION TESTS..."
    exit 1
else
    echo "WILL RUN TESTS WITH SPARK_HOME: ${SPARK_HOME}"
    # Spark 3.1.1 includes https://github.com/apache/spark/pull/31540
    # which helps with spurious task failures as observed in our tests. If you are running
    # Spark versions before 3.1.1, this sets the spark.max.taskFailures to 4 to allow for
    # more lineant configuration, else it will set them to 1 as spurious task failures are not expected
    # for Spark 3.1.1+
    VERSION_STRING=`$SPARK_HOME/bin/pyspark --version 2>&1|grep -v Scala|awk '/version\ [0-9.]+/{print $NF}'`
    VERSION_STRING="${VERSION_STRING/-SNAPSHOT/}"
    [[ -z $VERSION_STRING ]] && { echo "Unable to detect the Spark version at $SPARK_HOME"; exit 1; }
    [[ -z $SPARK_SHIM_VER ]] && { SPARK_SHIM_VER="spark${VERSION_STRING//./}"; }

    echo "Detected Spark version $VERSION_STRING (shim version: $SPARK_SHIM_VER)"

    PLUGIN_JARS=$(echo "$SCRIPTPATH"/target/dependency/rapids-4-spark*.jar)
    UDF_EXAMPLE_JARS=$(echo "$SCRIPTPATH"/target/rapids-4-spark-udf-examples*.jar)
    ALL_JARS="$PLUGIN_JARS $UDF_EXAMPLE_JARS"
    echo "AND PLUGIN JARS: $ALL_JARS"

    RUN_TESTS_COMMAND=("$SCRIPTPATH"/runtests.py
      --rootdir
      "$SCRIPTPATH"
      "$SCRIPTPATH"/src/main/python)

    # --ignore=target is used to exclude the target directory which contains unrelated python files.
    TEST_COMMON_OPTS=(-v
          -rfExXs
          "$TEST_ARGS"
          --color=yes
          --ignore=target
          "$@")

    "$SPARK_HOME"/bin/spark-submit --jars "${ALL_JARS// /,}" \
        --master local[1] \
        "${RUN_TESTS_COMMAND[@]}" "${TEST_COMMON_OPTS[@]}"

fi

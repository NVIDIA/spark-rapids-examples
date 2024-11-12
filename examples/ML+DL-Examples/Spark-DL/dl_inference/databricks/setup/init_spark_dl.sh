#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -x
# spark-rapids plugin
SCALA_VERSION=2.12
SPARK_RAPIDS_VERSION=24.10.0
RAPIDS_JAR=rapids-4-spark_${SCALA_VERSION}-${SPARK_RAPIDS_VERSION}.jar
curl -L https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_${SCALA_VERSION}/${SPARK_RAPIDS_VERSION}/${RAPIDS_JAR} -o \
    /databricks/jars/${RAPIDS_JAR}

# setup cuda: install cudatoolkit 11.8 via runfile approach
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit
# reset symlink and update library loading paths
rm /usr/local/cuda
ln -s /usr/local/cuda-11.8 /usr/local/cuda

# install requirements
sudo /databricks/python3/bin/pip3 install --upgrade pip
sudo /databricks/python3/bin/pip3 install --upgrade --force-reinstall -r REQ_PATH

set +x

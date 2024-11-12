#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

if [[ -z $SPARK_DL_HOME ]]; then
    echo "Error: Please set SPARK_DL_HOME per README.md"
    exit 1
fi

echo "**** copying requirements.txt to gs://${SPARK_DL_HOME}/requirements.txt ****"
gcloud storage cp requirements.txt gs://${SPARK_DL_HOME}/requirements.txt

echo "**** copying init_spark_dl.sh to gs://${SPARK_DL_HOME}/init_spark_dl.sh ****"
gcloud storage cp init_spark_dl.sh gs://${SPARK_DL_HOME}/init_spark_dl.sh

echo "**** copying conditional_generation.ipynb to gs://${SPARK_DL_HOME}/conditional_generation.ipynb ****"
gcloud storage cp ../conditional_generation.ipynb gs://${SPARK_DL_HOME}/conditional_generation.ipynb

echo "**** copying spark-rapids.sh to gs://${SPARK_DL_HOME}/spark-rapids.sh ****"
curl -LO https://raw.githubusercontent.com/GoogleCloudDataproc/initialization-actions/master/spark-rapids/spark-rapids.sh
gcloud storage cp spark-rapids.sh gs://${SPARK_DL_HOME}/spark-rapids.sh
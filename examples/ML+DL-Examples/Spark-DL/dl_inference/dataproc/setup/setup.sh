#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

if [[ -z $SPARK_DL_HOME ]]; then
    echo "Error: Please set SPARK_DL_HOME per README.md"
    exit 1
fi

gcloud storage cp requirements.txt gs://${SPARK_DL_HOME}/requirements.txt
gcloud storage cp init_spark_dl.sh gs://${SPARK_DL_HOME}/init_spark_dl.sh
gcloud storage cp ../conditional_generation.ipynb gs://${SPARK_DL_HOME}/conditional_generation.ipynb
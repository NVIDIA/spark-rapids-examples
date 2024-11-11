#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

if [ -z "$INIT_PATH" ] || [ -z "$NOTEBOOK_PATH" ] || [ -z "$REQ_PATH" ]; then
    echo "Error: One or more required paths are not set."
    echo "Please ensure INIT_PATH, NOTEBOOK_PATH, and REQ_PATH are all defined."
    exit 1
fi

if [[ "$REQ_PATH" != dbfs:/* ]]; then
    echo "Error: Make sure REQ_PATH begins with 'dbfs:/'."
    exit 1
fi

databricks fs cp requirements.txt $REQ_PATH
NEW_REQ_PATH="/dbfs${REQ_PATH#dbfs:}"
sed -i "s|REQ_PATH|$NEW_REQ_PATH|" init_spark_dl.sh
databricks workspace import $INIT_PATH --format AUTO --file init_spark_dl.sh
databricks workspace import $NOTEBOOK_PATH --format JUPYTER --file conditional_generation.ipynb
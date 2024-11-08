# Spark DL Inference on Databricks

Distributed deep learning inference using the PySpark [predict_batch_udf](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.functions.predict_batch_udf.html#pyspark.ml.functions.predict_batch_udf) function on Databricks. 

## Setup

1. Install latest [databricks-cli](https://docs.databricks.com/en/dev-tools/cli/tutorial.html) and configure for your workspace.

2. Set your paths:
    ```shell
    export INIT_PATH=/path/in/workspace/to/init_spark_dl_torch.sh
    export NOTEBOOK_PATH=/path/in/workspace/to/conditional_generation.ipynb
    export REQ_PATH=/path/to/requirements.txt # excluding the 'dbfs:' header
    ```
    Copy files to Databricks:
    ```
    databricks fs cp requirements.txt dbfs:$REQ_PATH
    sed -i "s|/REQ_PATH|$REQ_PATH|" init_spark_dl_torch.sh
    databricks workspace import $INIT_PATH --format AUTO --file init_spark_dl_torch.sh
    databricks workspace import $NOTEBOOK_PATH --format JUPYTER --file conditional_generation.ipynb
    ```

3. Launch cluster with init script.
    - Go to `Compute > Create compute` and set the desired cluster settings.
        - Integration with Triton inference server uses stage-level scheudling. Make sure to:
            - use a cluster with GPU resources
            - set a value for `spark.executor.cores`
            - ensure that `spark.executor.resource.gpu.amount` <= 1
    - Under `Advanced Options > Init Scripts`, upload the init script from your workspace.

4. Run the notebook interactively from the workspace.

### Triton Inference Server

The examples also demonstrate integration with [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server), an open-source, GPU-accelerated serving solution for DL. 

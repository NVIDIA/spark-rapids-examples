# Spark DL Inference on Databricks

Distributed deep learning inference using the PySpark [predict_batch_udf](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.functions.predict_batch_udf.html#pyspark.ml.functions.predict_batch_udf) function on Databricks.  
The examples also demonstrate integration with [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server), an open-source, GPU-accelerated serving solution for DL. 

## Setup

1. Install latest [databricks-cli](https://docs.databricks.com/en/dev-tools/cli/tutorial.html) and configure for your workspace.

2. Specify the destination filepaths on Databricks:
    The notebook and init script will be imported to your workspace, and the requirements to DBFS.
    ```shell
    export NOTEBOOK_PATH=</path/in/workspace/to/conditional_generation.ipynb>
    export INIT_PATH=</path/in/workspace/to/init_spark_dl.sh>
    export REQ_PATH=<dbfs:/path/to/requirements.txt>
    ```

3. Run the setup script, which will copy files to Databricks: 
    ```shell
    cd setup
    chmod +x setup.sh
    ./setup.sh
    ```

4. Launch the cluster with the provided script (8 node GPU cluster):
    ```shell
    chmod +x start_cluster.sh
    ./start_cluster.sh
    ```

    OR, start the cluster from the Databricks UI:  

    - Go to `Compute > Create compute` and set the desired cluster settings.
        - Integration with Triton inference server uses stage-level scheduling. Make sure to:
            - use a cluster with GPU resources
            - set a value for `spark.executor.cores`
            - ensure that `spark.executor.resource.gpu.amount` = 1
    - Under `Advanced Options > Init Scripts`, upload the init script from your workspace.

5. Run the notebook interactively from the workspace.

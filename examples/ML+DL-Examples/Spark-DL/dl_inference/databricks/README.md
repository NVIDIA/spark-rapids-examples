# Spark DL Inference on Databricks

**Note**: fields in \<brackets\> require user inputs.

## Setup

1. Install the latest [databricks-cli](https://docs.databricks.com/en/dev-tools/cli/tutorial.html) and configure for your workspace.

2. Specify the path to the notebook and init script, and the destination filepaths on Databricks.
    As an example for a PyTorch notebook:
    ```shell
    export NOTEBOOK_SRC=</path/to/notebook_torch.ipynb>
    export NOTEBOOK_DEST=</Users/someone@example.com/spark-dl/notebook_torch.ipynb>

    export INIT_SRC=$(pwd)/setup/init_spark_dl.sh
    export INIT_DEST=</Users/someone@example.com/spark-dl/init_spark_dl.sh>
    ```
3. Specify the framework to torch or tf, corresponding to the notebook you wish to run. Continuing with the PyTorch example:
    ```shell
    export FRAMEWORK=torch
    ```
    This will tell the init script which libraries to install on the cluster.

4. Copy the files to the Databricks Workspace:
    ```shell
    databricks workspace import $INIT_DEST --format AUTO --file $INIT_SRC
    databricks workspace import $NOTEBOOK_DEST --format JUPYTER --file $NOTEBOOK_SRC
    ```

5. Launch the cluster with the provided script (note that the script specifies **Azure instances** by default; change as needed):
    ```shell
    cd setup
    chmod +x start_cluster.sh
    ./start_cluster.sh
    ```

    OR, start the cluster from the Databricks UI:  

    - Go to `Compute > Create compute` and set the desired cluster settings.
        - Integration with Triton inference server uses stage-level scheduling (Spark>=3.4.0). Make sure to:
            - use a cluster with GPU resources
            - set a value for `spark.executor.cores`
            - ensure that `spark.executor.resource.gpu.amount` = 1
    - Under `Advanced Options > Init Scripts`, upload the init script from your workspace.
    - Under environment variables, set `FRAMEWORK=torch` or `FRAMEWORK=tf` based on the notebook used.
    - For Tensorflow notebooks, we recommend setting the environment variable `TF_GPU_ALLOCATOR=cuda_malloc_async` (especially for Huggingface LLM models), which enables the CUDA driver to implicity release unused memory from the pool. 

6. Navigate to the notebook in your workspace and attach it to the cluster. The default cluster name is `spark-dl-$FRAMEWORK`.  
# Spark DL Inference on Databricks

## Setup

1. Install the latest [databricks-cli](https://docs.databricks.com/en/dev-tools/cli/tutorial.html) and configure for your workspace.

2. Specify the path to the notebook and init script (_torch or _tf), and the destination filepaths on Databricks.
    As an example for a PyTorch notebook:
    ```shell
    export NOTEBOOK_SRC=/path/to/notebook_torch.ipynb
    export NOTEBOOK_DEST=/Users/someone@example.com/spark-dl/notebook_torch.ipynb

    export INIT_SRC=/path/to/setup/init_spark_dl_torch.sh
    export INIT_DEST=/Users/someone@example.com/spark-dl/init_spark_dl_torch.sh
    ```

3. Copy the files to the Databricks Workspace:
    ```shell
    databricks workspace import $INIT_DEST --format AUTO --file $INIT_SRC
    databricks workspace import $NOTEBOOK_DEST --format JUPYTER --file $NOTEBOOK_SRC
    ```

4. Launch the cluster with the provided script (note that the script specifies **Azure instances** by default; change as needed):
    ```shell
    export CLUSTER_NAME=spark-dl-inference-torch
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
    - For Tensorflow notebooks, we recommend setting the environment variable `TF_GPU_ALLOCATOR=cuda_malloc_async` (especially for Huggingface LLM models), which enables the CUDA driver to implicity release unused memory from the pool. 

5. Navigate to the notebook in your workspace and attach it to the cluster.  
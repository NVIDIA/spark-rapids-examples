# Spark DL Inference on Databricks

**Note**: fields in \<brackets\> require user inputs.

## Setup

1. Install the latest [databricks-cli](https://docs.databricks.com/en/dev-tools/cli/tutorial.html) and configure for your workspace.

2. Specify the path to your Databricks workspace:
    ```shell
    export WS_PATH=</Users/someone@example.com>

    export NOTEBOOK_DEST=${WS_PATH}/spark-dl/notebook_torch.ipynb
    export UTILS_DEST=${WS_PATH}/spark-dl/pytriton_utils.py
    export INIT_DEST=${WS_PATH}/spark-dl/init_spark_dl.sh
    ```
3. Specify the local paths to the notebook you wish to run, the utils file, and the init script.
    As an example for a PyTorch notebook:
    ```shell
    export NOTEBOOK_SRC=</path/to/notebook_torch.ipynb>
    export UTILS_SRC=</path/to/pytriton_utils.py>
    export INIT_SRC=$(pwd)/setup/init_spark_dl.sh
    ```
4. Specify the framework to torch or tf, corresponding to the notebook you wish to run. Continuing with the PyTorch example:
    ```shell
    export FRAMEWORK=torch
    ```
    This will tell the init script which libraries to install on the cluster.

5. Copy the files to the Databricks Workspace:
    ```shell
    databricks workspace import $NOTEBOOK_DEST --format JUPYTER --file $NOTEBOOK_SRC
    databricks workspace import $UTILS_DEST --format AUTO --file $UTILS_SRC
    databricks workspace import $INIT_DEST --format AUTO --file $INIT_SRC
    ```

6. Launch the cluster with the provided script.
**Note:** The LLM examples (e.g. deepseek-r1, gemma-7b) require greater GPU RAM (>18GB). For these notebooks, we recommend modifying the startup script node types to use A10 GPU instances. Note that the script specifies **Azure instances** by default; change as needed. 
    ```shell
    cd setup
    chmod +x start_cluster.sh
    ./start_cluster.sh
    ```
    OR, start the cluster from the Databricks UI:  

    - Go to `Compute > Create compute` and set the desired cluster settings.
        - Integration with Triton inference server uses stage-level scheduling (Spark>=3.4.0). Make sure to:
            - use a cluster with GPU resources (for LLM examples, make sure you have sufficient GPU RAM)
            - set a value for `spark.executor.cores`
            - ensure that `spark.executor.resource.gpu.amount` = 1
    - Under `Advanced Options > Init Scripts`, upload the init script from your workspace.
    - Under environment variables, set:
        - `FRAMEWORK=torch` or `FRAMEWORK=tf` based on the notebook used.
        - `HF_HOME=/dbfs/FileStore/hf_home` to cache Huggingface models in DBFS.
        - `TF_GPU_ALLOCATOR=cuda_malloc_async` to implicity release unused GPU memory in Tensorflow notebooks.

    

7. Navigate to the notebook in your workspace and attach it to the cluster. The default cluster name is `spark-dl-inference-$FRAMEWORK`.  
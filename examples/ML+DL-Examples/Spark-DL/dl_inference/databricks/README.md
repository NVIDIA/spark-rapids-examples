# Spark DL Inference on Databricks

**Note**: fields in \<brackets\> require user inputs.  
Make sure you are in [this](./) directory.

## Setup

1. Install the latest [databricks-cli](https://docs.databricks.com/en/dev-tools/cli/tutorial.html) and configure for your workspace.

2. Specify the path to your Databricks workspace:
    ```shell
    export WS_PATH=</Users/someone@example.com>
    ```

    ```shell
    export SPARK_DL_WS=${WS_PATH}/spark-dl
    databricks workspace mkdirs ${SPARK_DL_WS}
    ```
3. Specify the local paths to the notebook you wish to run, the utils file, and the init script.
    As an example for a PyTorch notebook:
    ```shell
    export NOTEBOOK_SRC=</path/to/notebook_torch.ipynb>
    ```
    ```shell
    export UTILS_SRC=$(realpath ../pytriton_utils.py)
    export INIT_SRC=$(pwd)/setup/init_spark_dl.sh
    ```
4. Specify the framework to torch, tf, or vllm, corresponding to the notebook you wish to run. Continuing with the PyTorch example:
    ```shell
    export FRAMEWORK=torch
    ```
    This will tell the init script which libraries to install on the cluster.

5. Copy the files to the Databricks Workspace:
    ```shell
    databricks workspace import ${SPARK_DL_WS}/notebook_torch.ipynb --format JUPYTER --file $NOTEBOOK_SRC
    databricks workspace import ${SPARK_DL_WS}/pytriton_utils.py --format AUTO --file $UTILS_SRC
    databricks workspace import ${SPARK_DL_WS}/init_spark_dl.sh --format AUTO --file $INIT_SRC
    ```

6. Launch the cluster with the provided script. By default the script will create a cluster with 2 A10 worker nodes and 1 A10 driver node. For vLLM examples, the worker nodes will have 2 GPUs each to demo tensor parallelism. (Note that the script uses **Azure instances** by default; change as needed).
    ```shell
    cd setup
    chmod +x start_cluster.sh
    ./start_cluster.sh
    ```
    OR, start the cluster from the Databricks UI:  

    - Go to `Compute > Create compute` and set the desired cluster settings.
        - Integration with Triton inference server uses stage-level scheduling (Spark>=3.4.0). Make sure to:
            - use GPU instances (we recommend A10/L4 for sufficient VRAM)
            - set `spark.executor.cores` = cores per node
            - set `spark.executor.resource.gpu.amount` = 1 for torch/tf, or = 2 for vllm
    - Under `Advanced Options > Init Scripts`, upload the init script from your workspace.
    - Under environment variables, set:
        - `FRAMEWORK=torch`, `FRAMEWORK=tf`, or `FRAMEWORK=vllm` based on the notebook used.
        - `TF_GPU_ALLOCATOR=cuda_malloc_async` to implicity release unused GPU memory in Tensorflow notebooks.

    

7. Navigate to the notebook in your workspace and attach it to the cluster. The default cluster name is `spark-dl-inference-$FRAMEWORK`.  
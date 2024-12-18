# Spark DL Inference on Databricks

TODO: fix instructions (Any of the example notebooks can be run on databricks... set the dest and src paths, set the right requirements, etc. )

## Setup

1. Install latest [databricks-cli](https://docs.databricks.com/en/dev-tools/cli/tutorial.html) and configure for your workspace.

2. Specify the path to the notebook and init script (torch or tf), and the destination filepaths on Databricks:
    The notebook and init script will be imported to your workspace, and the requirements to DBFS. As an example for a torch notebook:
    ```shell
    export NOTEBOOK_SRC=/path/to/notebook_torch.ipynb
    export NOTEBOOK_DEST=/Users/someone@example.com/spark-dl/notebook_torch.ipynb

    export INIT_SRC=/path/to/init_spark_dl_torch.sh
    export INIT_DEST=/Users/someone@example.com/spark-dl/init_spark_dl_torch.sh
    ```

3. Copy the files to Databricks:
    ```shell
    databricks workspace import $INIT_DEST --format AUTO --file $INIT_SRC
    databricks workspace import $NOTEBOOK_DEST --format JUPYTER --file $NOTEBOOK_SRC
    ```

4. Launch the cluster with the provided script (defaults to 4 node GPU cluster):
    ```shell
    export CLUSTER_NAME=spark-dl-inference-torch
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

5. Navigate to the notebook in your Databricks workspace. Attach the notebook to the cluster and run the cells interactively.  

## Inference with PyTriton 

<img src="../images/spark-pytriton.png" alt="drawing" width="1000"/>

The diagram above demonstrates how Spark distributes inference tasks to run on the [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server), with PyTriton handling request/response communication with the server.  

The process looks like this:
- Distribute a PyTriton task across the Spark cluster, instructing each node to launch a Triton server process.
    - Use stage-level scheduling to ensure each node is assigned a single startup task.
- Define a Triton inference function, which contains a client that binds to the local server on a given node and sends inference requests.
- Wrap the Triton inference function in a predict_batch_udf to launch parallel inference requests using Spark.
- Finally, distribute a shutdown signal to terminate the Triton server processes on each node.

For more information, see the [PyTriton docs](https://triton-inference-server.github.io/pytriton/latest/high_level_design/).
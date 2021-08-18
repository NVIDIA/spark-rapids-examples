Get Started with XGBoost4J-Spark on Databricks
======================================================

This is a getting started guide to XGBoost4J-Spark on Databricks. At the end of this guide, the reader will be able to run a sample Apache Spark application that runs on NVIDIA GPUs on Databricks.

Prerequisites
-------------

* Apache Spark 3.0.1+ running in DataBricks Runtime 7.3 ML with GPU. Make sure it matches the hardware and software requirements below.
* Hardware Requirements
  * NVIDIA Pascal™ GPU architecture or better
  * Multi-node clusters with homogenous GPU configuration
* Software Requirements
  * Ubuntu 18.04, 20.04/CentOS7, CentOS8
  * CUDA 11.0-11.4
  * NVIDIA driver compatible with your CUDA
  * NCCL 2.7.8

The number of GPUs per node dictates the number of Spark executors that can run in that node. Each executor should only be allowed to run 1 task at any given time.

Start A Databricks Cluster
--------------------------

Create a Databricks cluster (`Clusters` -> `+ Create Cluster`) that meets the above prerequisites.

1. Make sure to use the 7.0 ML with GPU Databricks runtime.
2. Use nodes with 1 GPU each such as p3.xlarge or Standard\_NC6s\_v3. We currently don't support nodes with multiple GPUs.  p2 (AWS) and NC12/24 (Azure) nodes do not meet the architecture requirements for the XGBoost worker (although they can be used for the driver node).  
3. Under Autopilot Options, disable autoscaling.
4. Choose the number of workers that matches the number of GPUs you want to use.
5. Select a worker type that has 1 GPU for the worker like p3.xlarge or NC6s_v3, for example.

After you start a Databricks cluster, use the initialization notebooks -- [7.0 notebook](/docs/get-started/xgboost-examples/csp/databricks/init-notebook-for-rapids-spark-xgboost-on-databricks-gpu-7.0-ml.ipynb
) to setup execution.

The initialization notebooks will perform the following steps:

1. Downloading the CUDA, Rapids-4-spark and Rapids XGBoost4j Spark jars
    * [*cudf-latest.jar*](https://repo1.maven.org/maven2/ai/rapids/cudf/21.08.2/)
    * [*xgboost4j-latest.jar*](https://repo1.maven.org/maven2/com/nvidia/xgboost4j_3.0/1.4.2-0.1.0/)
    * [*xgboost4j-spark-latest.jar*](https://repo1.maven.org/maven2/com/nvidia/xgboost4j-spark_3.0/1.4.2-0.1.0/)
    * [*rapids-latest.jar*](https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/21.08.0/)
2. Creating a new directory for initialization script in Databricks file system (DBFS)
3. Creating an initialization script inside the new directory to copy jars inside Databricks jar directory
4. Download and decompress the Sample Mortgage Notebook dataset

After executing the steps in the initialization notebook, please follow the 1. Cluster initialization script and 2. Install the xgboost4j_spark jar in the cluster to ensure it is ready for XGBoost training.

Add cluster initialization script and Spark Configs
---------------------------

1. See [Initialization scripts](https://docs.databricks.com/user-guide/clusters/init-scripts.html) for how to configure cluster initialization scripts.
2. Edit your cluster, adding an initialization script from dbfs:/databricks/init_scripts/init.sh in the "Advanced Options" under "Init Scripts" tab
3. Now select the “Spark” tab, and paste the following config options into the Spark Config section. Change the config values based on the workers you choose.

    ``` bash
    spark.plugins com.nvidia.spark.SQLPlugin
    spark.rapids.memory.gpu.pooling.enabled false
    ```

4. Reboot the cluster

Install the xgboost4j_spark jar in the cluster
---------------------------

1. See [Libraries](https://docs.databricks.com/user-guide/libraries.html) for how to install jars from DBFS
2. Go to "Libraries" tab under your cluster and install dbfs:/FileStore/jars/${XGBOOST4J_SPARK_JAR} in your cluster by selecting the "DBFS" option for installing jars

These steps will ensure you are able to import xgboost libraries in python notebooks.

Import the GPU Mortgage Example Notebook
---------------------------

1. See [Managing Notebooks](https://docs.databricks.com/user-guide/notebooks/notebook-manage.html) on how to import a notebook.
2. Import the example notebook: [XGBoost4j-Spark mortgage notebook](/examples/mortgage/notebooks/scala/mortgage-gpu.ipynb)
3. Inside the mortgage example notebook, update the data paths from 
"/data/datasets/mortgage-small/train" to "dbfs:/FileStore/tables/mortgage/csv/train/mortgage_train_merged.csv"
"/data/datasets/mortgage-small/eval" to "dbfs:/FileStore/tables/mortgage/csv/test/mortgage_eval_merged.csv"

The example notebook comes with the following configuration, you can adjust this according to your setup.
See supported configuration options here: [xgboost parameters](/examples/app-parameters/supported_xgboost_parameters_python.md)

``` bash
params = { 
    'eta': 0.1,
    'gamma': 0.1,
    'missing': 0.0,
    'treeMethod': 'gpu_hist',
    'maxDepth': 10, 
    'maxLeaves': 256,
    'growPolicy': 'depthwise',
    'minChildWeight': 30.0,
    'lambda_': 1.0,
    'scalePosWeight': 2.0,
    'subsample': 1.0,
    'nthread': 1,
    'numRound': 100,
    'numWorkers': 1,
}
```

4. Run all the cells in the notebook.

5. View the results
In the cell 5 (Training), 7 (Transforming) and 8 (Accuracy of Evaluation) you will see the output.

```
--------------
==> Benchmark: 
Training takes 6.48 seconds
--------------

--------------
==> Benchmark: Transformation takes 3.2 seconds

--------------

------Accuracy of Evaluation------
Accuracy is 0.9980699597729774

```

<sup>*</sup> The timings in this Getting Started guide are only illustrative. Please see our [release announcement](https://medium.com/rapids-ai/nvidia-gpus-and-apache-spark-one-step-closer-2d99e37ac8fd) for official benchmarks.

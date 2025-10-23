Get Started with XGBoost4J-Spark on an Apache Spark Standalone Cluster
======================================================================
This is a getting started guide to XGBoost4J-Spark on an Apache Spark 3.2+ Standalone Cluster.
At the end of this guide, the user can run a sample Apache Spark Python application that runs on NVIDIA GPUs.

Prerequisites
-------------

* Apache Spark 3.2.0+ (e.g.: Spark 3.2.0)
* Hardware Requirements
  * NVIDIA Pascal™ GPU architecture or better
  * Multi-node clusters with homogenous GPU configuration
* Software Requirements
  * Ubuntu 20.04, 22.04/CentOS7, Rocky Linux 8
  * CUDA 11.5+
  * NVIDIA driver compatible with your CUDA
  * NCCL 2.7.8+
  * Python 3.8 or 3.9
  * NumPy
  * XGBoost 1.7.0+
  * cudf-cu11  

The number of GPUs in each host dictates the number of Spark executors that can run there.
Additionally, cores per Spark executor and cores per Spark task must match, such that each executor can run 1 task at any given time.

For example, if each host has 4 GPUs, there should be 4 or fewer executors running on each host,
and each executor should run at most 1 task (e.g.: a total of 4 tasks running on 4 GPUs).

In Spark Standalone mode, the default configuration is for an executor to take up all the cores assigned to each Spark Worker.
In this example, we will limit the number of cores to 1, to match our dataset.
Please see https://spark.apache.org/docs/latest/spark-standalone.html for more documentation regarding Standalone configuration.

We use `SPARK_HOME` environment variable to point to the Apache Spark cluster.
And here are the steps to enable the GPU resources discovery for Spark 3.2+.

1. Copy the spark config file from template

    ``` bash
    cd ${SPARK_HOME}/conf/
    cp spark-defaults.conf.template spark-defaults.conf
    ```

2. Add the following configs to the file `spark-defaults.conf`.

   The number in the first config should **NOT** be larger than the actual number of the GPUs on current host.
   This example uses 1 as below for one GPU on the host.

    ```bash
    spark.worker.resource.gpu.amount 1
    spark.worker.resource.gpu.discoveryScript ${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh
    ```
3. Install the XGBoost, cudf-cu11, numpy libraries on all nodes before running XGBoost application.

``` bash
pip install xgboost
pip install cudf-cu11 --extra-index-url=https://pypi.nvidia.com
pip install numpy
pip install scikit-learn
```

Get Application Files, Jar and Dataset
-------------------------------

Make sure you have prepared the necessary packages and dataset by following this [guide](../prepare-package-data/preparation-python.md)


#### Note: 
1. Mortgage and Taxi jobs have ETLs to generate the processed data.
2. For convenience, a subset of [Taxi](/datasets/) dataset is made available in this repo that can be readily used for launching XGBoost job. Use [ETL](standalone-python.md#launch-mortgage-or-taxi-etl-part) to generate larger datasets for training and testing.
3. Agaricus does not have an ETL process, it is combined with XGBoost as there is just a filter operation.


Launch a Standalone Spark Cluster
---------------------------------

1. Copy required jars to `$SPARK_HOME/jars` folder.

    ``` bash
    cp ${RAPIDS_JAR} $SPARK_HOME/jars/
    ```

2. Start the Spark Master process.

    ``` bash
    ${SPARK_HOME}/sbin/start-master.sh
    ```

    Note the hostname or ip address of the Master host, so that it can be given to each Worker process, in this example the Master and Worker will run on the same host.

3. Start a spark slave process.

    ``` bash
    export SPARK_MASTER=spark://`hostname -f`:7077
    export SPARK_CORES_PER_WORKER=1

    ${SPARK_HOME}/sbin/start-slave.sh ${SPARK_MASTER} -c ${SPARK_CORES_PER_WORKER}
    ```

    Note that in this example the Master and Worker processes are both running on the same host. This is not a requirement, as long as all hosts that are used to run the Spark app have access to the dataset.

Launch Mortgage or Taxi ETL Part
---------------------------
Use the ETL app to process raw Mortgage data. You can either use this ETLed data to split into training and evaluation data or run the ETL on different subsets of the dataset to produce training and evaluation datasets.

Note: For ETL jobs, Set `spark.task.resource.gpu.amount` to `1/spark.executor.cores`.
### ETL on GPU
``` bash
${SPARK_HOME}/bin/spark-submit \
    --master spark://$HOSTNAME:7077 \
    --executor-memory 32G \
    --conf spark.executor.resource.gpu.amount=1 \
    --conf spark.executor.cores=10 \
    --conf spark.task.resource.gpu.amount=0.1 \
    --conf spark.plugins=com.nvidia.spark.SQLPlugin \
    --conf spark.rapids.sql.incompatibleDateFormats.enabled=true \
    --conf spark.rapids.sql.csv.read.double.enabled=true \
    --conf spark.sql.cache.serializer=com.nvidia.spark.ParquetCachedBatchSerializer \
    --py-files ${SAMPLE_ZIP} \
    main.py \
    --mainClass='com.nvidia.spark.examples.mortgage.etl_main' \
    --format=csv \
    --dataPath="data::${SPARK_XGBOOST_DIR}/mortgage/input/" \
    --dataPath="out::${SPARK_XGBOOST_DIR}/mortgage/output/train/" \
    --dataPath="tmp::${SPARK_XGBOOST_DIR}/mortgage/output/tmp/"

# if generating eval data, change the data path to eval
# --dataPath="data::${SPARK_XGBOOST_DIR}/mortgage/input/"
# --dataPath="out::${SPARK_XGBOOST_DIR}/mortgage/output/eval/"
# --dataPath="tmp::${SPARK_XGBOOST_DIR}/mortgage/output/tmp/"
# if running Taxi ETL benchmark, change the class and data path params to
# -class com.nvidia.spark.examples.taxi.ETLMain  
# -dataPath="raw::${SPARK_XGBOOST_DIR}/taxi/your-path"
# -dataPath="out::${SPARK_XGBOOST_DIR}/taxi/your-path"
```
### ETL on CPU
```bash
${SPARK_HOME}/bin/spark-submit \
    --master spark://$HOSTNAME:7077 \
    --executor-memory 32G \
    --conf spark.executor.instances=1 \
    --py-files ${SAMPLE_ZIP} \
    main.py \
    --mainClass='com.nvidia.spark.examples.mortgage.etl_main' \
    --format=csv \
    --dataPath="data::${SPARK_XGBOOST_DIR}/mortgage/input/" \
    --dataPath="out::${SPARK_XGBOOST_DIR}/mortgage/output/train/" \
    --dataPath="tmp::${SPARK_XGBOOST_DIR}/mortgage/output/tmp/"

# if generating eval data, change the data path to eval
# --dataPath="data::${SPARK_XGBOOST_DIR}/mortgage/input/"
# --dataPath="out::${SPARK_XGBOOST_DIR}/mortgage/output/eval/"
# --dataPath="tmp::${SPARK_XGBOOST_DIR}/mortgage/output/tmp/"
# if running Taxi ETL benchmark, change the class and data path params to
# -class com.nvidia.spark.examples.taxi.ETLMain  
# -dataPath="raw::${SPARK_XGBOOST_DIR}/taxi/your-path"
# -dataPath="out::${SPARK_XGBOOST_DIR}/taxi/your-path"
```

Launch XGBoost Part on GPU
---------------------------

Variables required to run spark-submit command:

``` bash
# this is the same master host we defined while launching the cluster
export SPARK_MASTER=spark://`hostname -f`:7077

# Currently the number of tasks and executors must match the number of input files.
# For this example, we will set these such that we have 1 executor, with 1 core per executor

## take up the the whole worker
export SPARK_CORES_PER_EXECUTOR=${SPARK_CORES_PER_WORKER}

## run 1 executor
export SPARK_NUM_EXECUTORS=1

## cores/executor * num_executors, which in this case is also 1, limits
## the number of cores given to the application
export TOTAL_CORES=$((SPARK_CORES_PER_EXECUTOR * SPARK_NUM_EXECUTORS))

# spark driver memory
export SPARK_DRIVER_MEMORY=4g

# spark executor memory
export SPARK_EXECUTOR_MEMORY=8g

# example class to use
export EXAMPLE_CLASS=com.nvidia.spark.examples.mortgage.main
# or change to com.nvidia.spark.examples.taxi.main to run Taxi Xgboost benchmark
# or change to com.nvidia.spark.examples.agaricus.main to run Agaricus Xgboost benchmark

# tree construction algorithm
export TREE_METHOD=gpu_hist

# if you enable archive python environment
export PYSPARK_DRIVER_PYTHON=python
export PYSPARK_PYTHON=./environment/bin/python
```

Run spark-submit:

``` bash
${SPARK_HOME}/bin/spark-submit                                                  \
 --conf spark.plugins=com.nvidia.spark.SQLPlugin                       \
 --conf spark.rapids.memory.gpu.pool=NONE                     \
 --conf spark.executor.resource.gpu.amount=1                           \
 --conf spark.task.resource.gpu.amount=1                              \
 --master ${SPARK_MASTER}                                                       \
 --driver-memory ${SPARK_DRIVER_MEMORY}                                         \
 --executor-memory ${SPARK_EXECUTOR_MEMORY}                                     \
 --conf spark.cores.max=${TOTAL_CORES}                                          \
 --archives your_pyspark_venv.tar.gz#environment     #if you enabled archive python environment \
 --jars ${RAPIDS_JAR}    \
 --py-files ${SAMPLE_ZIP}                   \
 ${MAIN_PY}                                                     \
 --mainClass=${EXAMPLE_CLASS}                                                   \
 --dataPath=train::${SPARK_XGBOOST_DIR}/mortgage/output/train/      \
 --dataPath=trans::${SPARK_XGBOOST_DIR}/mortgage/output/eval/      \
 --format=parquet                                 \
 --numWorkers=${SPARK_NUM_EXECUTORS}                                            \
 --treeMethod=${TREE_METHOD}                                                    \
 --numRound=100                                                                 \
 --maxDepth=8

 # Change the format to csv if your input file is CSV format.
 # Please make sure to change the class and data path while running Taxi or Agaricus benchmark  
```

In the `stdout` log on driver side, you should see timings<sup>*</sup> (in seconds), and the accuracy metric:

```
----------------------------------------------------------------------------------------------------
Training takes 14.65 seconds

----------------------------------------------------------------------------------------------------
Transformation takes 12.21 seconds

----------------------------------------------------------------------------------------------------
Accuracy is 0.9873692247091792
```

Launch XGBoost Part on CPU
---------------------------

If you are running this example after running the GPU example above, please set these variables,
to set both training and testing to run on the CPU exclusively:

``` bash
# this is the same master host we defined while launching the cluster
export SPARK_MASTER=spark://`hostname -f`:7077

# Currently the number of tasks and executors must match the number of input files.
# For this example, we will set these such that we have 1 executor, with 1 core per executor

## take up the the whole worker
export SPARK_CORES_PER_EXECUTOR=${SPARK_CORES_PER_WORKER}

## run 1 executor
export SPARK_NUM_EXECUTORS=1

## cores/executor * num_executors, which in this case is also 1, limits
## the number of cores given to the application
export TOTAL_CORES=$((SPARK_CORES_PER_EXECUTOR * SPARK_NUM_EXECUTORS))

# spark driver memory
export SPARK_DRIVER_MEMORY=4g

# spark executor memory
export SPARK_EXECUTOR_MEMORY=8g

# example class to use
export EXAMPLE_CLASS=com.nvidia.spark.examples.mortgage.main
# Please make sure to change the class while running Taxi or Agaricus benchmark    

# tree construction algorithm
export TREE_METHOD=hist

# if you enable archive python environment
export PYSPARK_DRIVER_PYTHON=python
export PYSPARK_PYTHON=./environment/bin/python
```

This is the same command as for the GPU example, repeated for convenience:

``` bash
${SPARK_HOME}/bin/spark-submit                                                  \
 --master ${SPARK_MASTER}                                                       \
 --driver-memory ${SPARK_DRIVER_MEMORY}                                         \
 --executor-memory ${SPARK_EXECUTOR_MEMORY}                                     \
 --conf spark.cores.max=${TOTAL_CORES}                                          \
 --archives your_pyspark_venv.tar.gz#environment     #if you enabled archive python environment \
 --jars ${RAPIDS_JAR}     \
 --py-files ${SAMPLE_ZIP}                       \
 ${SPARK_PYTHON_ENTRYPOINT}                                                     \
 --mainClass=${EXAMPLE_CLASS}                                                   \
 --dataPath=train::${DATA_PATH}/mortgage/output/train/      \
 --dataPath=trans::${DATA_PATH}/mortgage/output/eval/         \
 --format=parquet                                                               \
 --numWorkers=${SPARK_NUM_EXECUTORS}                                            \
 --treeMethod=${TREE_METHOD}                                                    \
 --numRound=100                                                                 \
 --maxDepth=8

 # Change the format to csv if your input file is CSV format.
 # Please make sure to change the class and data path while running Taxi or Agaricus benchmark  
 
```

In the `stdout` log on driver side, you should see timings<sup>*</sup> (in seconds), and the accuracy metric:

```
----------------------------------------------------------------------------------------------------
Training takes 225.7 seconds

----------------------------------------------------------------------------------------------------
Transformation takes 36.26 seconds

----------------------------------------------------------------------------------------------------
Accuracy is 0.9873709530950067
```

<sup>*</sup> The timings in this Getting Started guide are only illustrative.
Please see our [release announcement](https://medium.com/rapids-ai/nvidia-gpus-and-apache-spark-one-step-closer-2d99e37ac8fd) for official benchmarks.

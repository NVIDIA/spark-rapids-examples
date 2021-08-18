Get Started with XGBoost4J-Spark on an Apache Spark Standalone Cluster
======================================================================
This is a getting started guide to XGBoost4J-Spark on an Apache Spark 3.0+ Standalone Cluster. At the end of this guide, the user can run a sample Apache Spark Python application that runs on NVIDIA GPUs.

Prerequisites
-------------

* Apache Spark 3.0.1+ Standalone Cluster (e.g.: Spark 3.0.1)
* Hardware Requirements
  * NVIDIA Pascal™ GPU architecture or better
  * Multi-node clusters with homogenous GPU configuration
* Software Requirements
  * Ubuntu 18.04, 20.04/CentOS7, CentOS8
  * CUDA 11.0-11.4
  * NVIDIA driver compatible with your CUDA
  * NCCL 2.7.8
  * Python 3.6+
  * NumPy

The number of GPUs in each host dictates the number of Spark executors that can run there. Additionally, cores per Spark executor and cores per Spark task must match, such that each executor can run 1 task at any given time.

For example, if each host has 4 GPUs, there should be 4 or less executors running on each host, and each executor should run at most 1 task (e.g.: a total of 4 tasks running on 4 GPUs).

In Spark Standalone mode, the default configuration is for an executor to take up all the cores assigned to each Spark Worker. In this example, we will limit the number of cores to 1, to match our dataset. Please see https://spark.apache.org/docs/latest/spark-standalone.html for more documentation regarding Standalone configuration.

We use `SPARK_HOME` environment variable to point to the cluster's Apache Spark cluster. And here are the steps to enable the GPU resources discovery for Spark 3.0+.

1. Copy the spark config file from template

    ``` bash
    cd ${SPARK_HOME}/conf/
    cp spark-defaults.conf.template spark-defaults.conf
    ```

2. Add the following configs to the file `spark-defaults.conf`.

   The number in the first config should **NOT** be larger than the actual number of the GPUs on current host. This example uses 1 as below for one GPU on the host.

    ```bash
    spark.worker.resource.gpu.amount 1
    spark.worker.resource.gpu.discoveryScript ${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh
    ```

Get Application Files, Jar and Dataset
-------------------------------

Make sure you have prepared the necessary packages and dataset by following this [guide](/docs/get-started/xgboost-examples/prepare-package-data/preparation-python.md)

Launch a Standalone Spark Cluster
---------------------------------

1. Copy required jars to `$SPARK_HOME/jars` folder

    ``` bash
    cp ${RAPIDS_JAR} $SPARK_HOME/jars/
    cp ${CUDF_JAR} $SPARK_HOME/jars/
    ```

2. Start the Spark Master process:

    ``` bash
    ${SPARK_HOME}/sbin/start-master.sh
    ```

    Note the hostname or ip address of the Master host, so that it can be given to each Worker process, in this example the Master and Worker will run on the same host.

3. Start a spark slave process:

    ``` bash
    export SPARK_MASTER=spark://`hostname -f`:7077
    export SPARK_CORES_PER_WORKER=1

    ${SPARK_HOME}/sbin/start-slave.sh ${SPARK_MASTER} -c ${SPARK_CORES_PER_WORKER}
    ```

    Note that in this example the Master and Worker processes are both running on the same host. This is not a requirement, as long as all hosts that are used to run the Spark app have access to the dataset.

Launch Mortgage ETL Example
---------------------------

Run spark-submit

``` bash
${SPARK_HOME}/bin/spark-submit \
    --master spark://$HOSTNAME:7077 \
    --executor-memory 32G \
    --conf spark.executor.resource.gpu.amount=1 \
    --conf spark.task.resource.gpu.amount=1 \
    --conf spark.plugins=com.nvidia.spark.SQLPlugin \
    --py-files ${SAMPLE_ZIP} \
    main.py \
    --mainClass='com.nvidia.spark.examples.mortgage.etl_main' \
    --format=csv \
    --dataPath="perf::${SPARK_XGBOOST_DIR}/mortgage/perf-train/" \
    --dataPath="acq::${SPARK_XGBOOST_DIR}/mortgage/acq-train/" \
    --dataPath="out::${SPARK_XGBOOST_DIR}/mortgage/out/train/"

# if generating eval data, change the data path to eval as well as the corresponding perf-eval and acq-eval data
# --dataPath="perf::${SPARK_XGBOOST_DIR}/mortgage/perf-eval"
# --dataPath="acq::${SPARK_XGBOOST_DIR}/mortgage/acq-eval"
# --dataPath="out::${SPARK_XGBOOST_DIR}/mortgage/out/eval/"
```

Launch GPU Mortgage Example
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
export EXAMPLE_CLASS=com.nvidia.spark.examples.mortgage.gpu_main

# tree construction algorithm
export TREE_METHOD=gpu_hist
```

Run spark-submit:

``` bash
${SPARK_HOME}/bin/spark-submit                                                  \
 --conf spark.plugins=com.nvidia.spark.SQLPlugin                       \
 --conf spark.rapids.memory.gpu.pooling.enabled=false                     \
 --conf spark.executor.resource.gpu.amount=1                           \
 --conf spark.task.resource.gpu.amount=1                              \
 --master ${SPARK_MASTER}                                                       \
 --driver-memory ${SPARK_DRIVER_MEMORY}                                         \
 --executor-memory ${SPARK_EXECUTOR_MEMORY}                                     \
 --conf spark.cores.max=${TOTAL_CORES}                                          \
 --jars ${CUDF_JAR},${RAPIDS_JAR},${XGBOOST4J_JAR},${XGBOOST4J_SPARK_JAR}     \
 --py-files ${XGBOOST4J_SPARK_JAR},${SAMPLE_ZIP}                   \
 ${MAIN_PY}                                                     \
 --mainClass=${EXAMPLE_CLASS}                                                   \
 --dataPath=train::${SPARK_XGBOOST_DIR}/mortgage/out/train/      \
 --dataPath=trans::${SPARK_XGBOOST_DIR}/mortgage/out/eval/      \
 --format=parquet                                 \
 --numWorkers=${SPARK_NUM_EXECUTORS}                                            \
 --treeMethod=${TREE_METHOD}                                                    \
 --numRound=100                                                                 \
 --maxDepth=8

 # Change the format to csv if your input file is CSV format.

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

Launch CPU Mortgage Example
---------------------------

If you are running this example after running the GPU example above, please set these variables, to set both training and testing to run on the CPU exclusively:

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
export EXAMPLE_CLASS=com.nvidia.spark.examples.mortgage.cpu_main

# tree construction algorithm
export TREE_METHOD=hist
```

This is the same command as for the GPU example, repeated for convenience:

``` bash
${SPARK_HOME}/bin/spark-submit                                                  \
 --master ${SPARK_MASTER}                                                       \
 --driver-memory ${SPARK_DRIVER_MEMORY}                                         \
 --executor-memory ${SPARK_EXECUTOR_MEMORY}                                     \
 --conf spark.cores.max=${TOTAL_CORES}                                          \
 --jars ${XGBOOST4J_JAR},${XGBOOST4J_SPARK_JAR}       \
 --py-files ${XGBOOST4J_SPARK_JAR},${SAMPLE_ZIP}                       \
 ${SPARK_PYTHON_ENTRYPOINT}                                                     \
 --mainClass=${EXAMPLE_CLASS}                                                   \
 --dataPath=train::${DATA_PATH}/mortgage/out/train/      \
 --dataPath=trans::${DATA_PATH}/mortgage/out/eval/         \
 --format=parquet                                                               \
 --numWorkers=${SPARK_NUM_EXECUTORS}                                            \
 --treeMethod=${TREE_METHOD}                                                    \
 --numRound=100                                                                 \
 --maxDepth=8

 # Change the format to csv if your input file is CSV format.
 
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

<sup>*</sup> The timings in this Getting Started guide are only illustrative. Please see our [release announcement](https://medium.com/rapids-ai/nvidia-gpus-and-apache-spark-one-step-closer-2d99e37ac8fd) for official benchmarks.

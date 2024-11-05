Get Started with XGBoost4J-Spark on Apache Hadoop YARN
======================================================

This is a getting started guide to XGBoost4J-Spark on Apache Hadoop YARN supporting GPU scheduling. 
At the end of this guide, the reader will be able to run a sample Apache Spark application that runs on NVIDIA GPUs.

Prerequisites
-------------

* Apache Spark 3.2.0+ running on YARN supporting GPU scheduling. (e.g.: Spark 3.2.0, Hadoop-Yarn 3.3.0)
* Hardware Requirements
  * NVIDIA Pascal™ GPU architecture or better
  * Multi-node clusters with homogenous GPU configuration
* Software Requirements
  * Ubuntu 20.04, 22.04/CentOS7, Rocky Linux 8
  * CUDA 11.0+
  * NVIDIA driver compatible with your CUDA
  * NCCL 2.7.8+

The number of GPUs per NodeManager dictates the number of Spark executors that can run in that NodeManager. 
Additionally, cores per Spark executor and cores per Spark task must match, such that each executor can run 1 task at any given time.

For example: if each NodeManager has 4 GPUs, there should be 4 or fewer executors running on each NodeManager, 
and each executor should run 1 task (e.g.: A total of 4 tasks running on 4 GPUs). In order to achieve this, 
you may need to adjust `spark.task.cpus` and `spark.executor.cores` to match (both set to 1 by default).
Additionally, we recommend adjusting `executor-memory` to divide host memory evenly amongst the number of GPUs in each NodeManager,
such that Spark will schedule as many executors as there are GPUs in each NodeManager.

We use `SPARK_HOME` environment variable to point to the Apache Spark cluster.
And as to how to enable GPU scheduling and isolation for Yarn, 
please refer to [here](https://hadoop.apache.org/docs/r3.1.0/hadoop-yarn/hadoop-yarn-site/UsingGpus.html).

Get Jars and Dataset
-------------------------------

Make sure you have prepared the necessary packages and dataset by following this [guide](/docs/get-started/xgboost-examples/prepare-package-data/preparation-scala.md)

#### Note: 
1. Mortgage and Taxi jobs have ETLs to generate the processed data.
2. For convenience, a subset of [Taxi](/datasets/) dataset is made available in this repo that can be readily used for launching XGBoost job. Use [ETL](#etl) to generate larger datasets for trainig and testing. 
3. Agaricus does not have an ETL process, it is combined with XGBoost as there is just a filter operation.

Create a directory in HDFS, and copy:

``` bash
[xgboost4j_spark]$ hadoop fs -mkdir /tmp/xgboost4j_spark
[xgboost4j_spark]$ hadoop fs -copyFromLocal ${SPARK_XGBOOST_DIR}/mortgage/* /tmp/xgboost4j_spark
```

<span id="etl">Launch Mortgage or Taxi ETL Part</span>
---------------------------

Use the ETL app to process raw Mortgage data. You can either use this ETLed data to split into training and evaluation data or run the ETL on different subsets of the dataset to produce training and evaluation datasets.

Note: For ETL jobs, Set `spark.task.resource.gpu.amount` to `1/spark.executor.cores`.


Run spark-submit

``` bash
${SPARK_HOME}/bin/spark-submit \
   --conf spark.plugins=com.nvidia.spark.SQLPlugin \
   --conf spark.executor.resource.gpu.amount=1 \
   --conf spark.executor.cores=10 \
   --conf spark.task.resource.gpu.amount=0.1 \
   --conf spark.rapids.sql.incompatibleDateFormats.enabled=true \
   --conf spark.rapids.sql.csv.read.double.enabled=true \
   --conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
   --conf spark.sql.cache.serializer=com.nvidia.spark.ParquetCachedBatchSerializer \
   --files $SPARK_HOME/examples/src/main/scripts/getGpusResources.sh \
   --jars ${RAPIDS_JAR}                                           \
   --master yarn                                                                  \
   --deploy-mode ${SPARK_DEPLOY_MODE}                                             \
   --num-executors ${SPARK_NUM_EXECUTORS}                                         \
   --driver-memory ${SPARK_DRIVER_MEMORY}                                         \
   --executor-memory ${SPARK_EXECUTOR_MEMORY}                                     \
   --class com.nvidia.spark.examples.mortgage.ETLMain  \
   $SAMPLE_JAR \
   -format=csv \
   -dataPath="data::${SPARK_XGBOOST_DIR}/mortgage/input/" \
   -dataPath="out::${SPARK_XGBOOST_DIR}/mortgage/output/train/" \
   -dataPath="tmp::${SPARK_XGBOOST_DIR}/mortgage/output/tmp/"

# if generating eval data, change the data path to eval 
# -dataPath="data::${SPARK_XGBOOST_DIR}/mortgage/input/"
# -dataPath="out::${SPARK_XGBOOST_DIR}/mortgage/output/eval/"
# -dataPath="tmp::${SPARK_XGBOOST_DIR}/mortgage/output/tmp/"
# if running Taxi ETL benchmark, change the class and data path params to
# -class com.nvidia.spark.examples.taxi.ETLMain  
# -dataPath="raw::${SPARK_XGBOOST_DIR}/taxi/your-path"
# -dataPath="out::${SPARK_XGBOOST_DIR}/taxi/your-path"
```

Launch XGBoost Part on GPU
---------------------------

Variables required to run spark-submit command:

``` bash
# location where data was downloaded 
export DATA_PATH=hdfs:/tmp/xgboost4j_spark/data

# spark deploy mode (see Apache Spark documentation for more information) 
export SPARK_DEPLOY_MODE=cluster

# run a single executor for this example to limit the number of spark tasks and
# partitions to 1 as currently this number must match the number of input files
export SPARK_NUM_EXECUTORS=1

# spark driver memory
export SPARK_DRIVER_MEMORY=4g

# spark executor memory
export SPARK_EXECUTOR_MEMORY=8g

# example class to use
export EXAMPLE_CLASS=com.nvidia.spark.examples.mortgage.Main
# or change to com.nvidia.spark.examples.taxi.Main to run Taxi Xgboost benchmark
# or change to com.nvidia.spark.examples.agaricus.Main to run Agaricus Xgboost benchmark

# tree construction algorithm
export TREE_METHOD=gpu_hist
```

Run spark-submit:

``` bash
${SPARK_HOME}/bin/spark-submit                                                  \
 --conf spark.plugins=com.nvidia.spark.SQLPlugin \
 --conf spark.rapids.memory.gpu.pool=NONE \
 --conf spark.executor.resource.gpu.amount=1 \
 --conf spark.task.resource.gpu.amount=1 \
 --conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
 --files $SPARK_HOME/examples/src/main/scripts/getGpusResources.sh \
 --jars ${RAPIDS_JAR}                                           \
 --master yarn                                                                  \
 --deploy-mode ${SPARK_DEPLOY_MODE}                                             \
 --num-executors ${SPARK_NUM_EXECUTORS}                                         \
 --driver-memory ${SPARK_DRIVER_MEMORY}                                         \
 --executor-memory ${SPARK_EXECUTOR_MEMORY}                                     \
 --class ${EXAMPLE_CLASS}                                                       \
 ${SAMPLE_JAR}                                                                 \
 -dataPath=train::${SPARK_XGBOOST_DIR}/mortgage/output/train/                   \
 -dataPath=trans::${SPARK_XGBOOST_DIR}/mortgage/output/eval/                    \
 -format=parquet                                                                \
 -numWorkers=${SPARK_NUM_EXECUTORS}                                             \
 -treeMethod=${TREE_METHOD}                                                     \
 -numRound=100                                                                  \
 -maxDepth=8                                                                    
  # Please make sure to change the class and data path while running Taxi or Agaricus benchmark   
```

In the `stdout` driver log, you should see timings<sup>*</sup> (in seconds), and the accuracy metric(take Mortgage as example):

```
--------------
==> Benchmark: Elapsed time for [Mortgage GPU train csv stub Unknown Unknown Unknown]: 29.642s
--------------

--------------
==> Benchmark: Elapsed time for [Mortgage GPU transform csv stub Unknown Unknown Unknown]: 21.272s
--------------

--------------
==> Benchmark: Accuracy for [Mortgage GPU Accuracy csv stub Unknown Unknown Unknown]: 0.9874184013493451
--------------
```

Launch XGBoost Part on CPU
---------------------------

If you are running this example after running the GPU example above, please set these variables, to set both training and testing to run on the CPU exclusively:

``` bash
# location where data was downloaded 
export DATA_PATH=hdfs:/tmp/xgboost4j_spark/data

# spark deploy mode (see Apache Spark documentation for more information) 
export SPARK_DEPLOY_MODE=cluster

# run a single executor for this example to limit the number of spark tasks and
# partitions to 1 as currently this number must match the number of input files
export SPARK_NUM_EXECUTORS=1

# spark driver memory
export SPARK_DRIVER_MEMORY=4g

# spark executor memory
export SPARK_EXECUTOR_MEMORY=8g

# example class to use
export EXAMPLE_CLASS=com.nvidia.spark.examples.mortgage.Main
# Please make sure to change the class while running Taxi or Agaricus benchmark   

# tree construction algorithm
export TREE_METHOD=hist
```

This is the same command as for the GPU example, repeated for convenience:

``` bash
${SPARK_HOME}/bin/spark-submit                                                  \
 --master yarn                                                                  \
 --deploy-mode ${SPARK_DEPLOY_MODE}                                             \
 --num-executors ${SPARK_NUM_EXECUTORS}                                         \
 --driver-memory ${SPARK_DRIVER_MEMORY}                                         \
 --executor-memory ${SPARK_EXECUTOR_MEMORY}                                     \
 --class ${EXAMPLE_CLASS}                                                       \
 ${SAMPLE_JAR}                                                                 \
 -dataPath=train::${SPARK_XGBOOST_DIR}/mortgage/output/train/                   \
 -dataPath=trans::${SPARK_XGBOOST_DIR}/mortgage/output/eval/                    \
 -format=parquet                                                                \
 -numWorkers=${SPARK_NUM_EXECUTORS}                                             \
 -treeMethod=${TREE_METHOD}                                                     \
 -numRound=100                                                                  \
 -maxDepth=8                            
   
  # Please make sure to change the class and data path while running Taxi or Agaricus benchmark                                                       
                                      
```

In the `stdout` driver log, you should see timings<sup>*</sup> (in seconds), and the accuracy metric(take Mortgage as example):

```
--------------
==> Benchmark: Elapsed time for [Mortgage CPU train csv stub Unknown Unknown Unknown]: 286.398s
--------------

--------------
==> Benchmark: Elapsed time for [Mortgage CPU transform csv stub Unknown Unknown Unknown]: 49.836s
--------------

--------------
==> Benchmark: Accuracy for [Mortgage CPU Accuracy csv stub Unknown Unknown Unknown]: 0.9873709530950067
--------------
```

<sup>*</sup> The timings in this Getting Started guide are only for illustrative purpose.
Please see our [release announcement](https://medium.com/rapids-ai/nvidia-gpus-and-apache-spark-one-step-closer-2d99e37ac8fd) for official benchmarks.

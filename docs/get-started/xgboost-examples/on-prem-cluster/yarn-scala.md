Get Started with XGBoost4J-Spark on Apache Hadoop YARN
======================================================

This is a getting started guide to XGBoost4J-Spark on Apache Hadoop YARN supporting GPU scheduling. 
At the end of this guide, the reader will be able to run a sample Apache Spark application that runs on NVIDIA GPUs.

Prerequisites
-------------

* Apache Spark 3.1.1+ running on YARN supporting GPU scheduling. (e.g.: Spark 3.1.1, Hadoop-Yarn 3.3.0)
* Hardware Requirements
  * NVIDIA Pascal™ GPU architecture or better
  * Multi-node clusters with homogenous GPU configuration
* Software Requirements
  * Ubuntu 18.04, 20.04/CentOS7, CentOS8
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

Create a directory in HDFS, and copy:

``` bash
[xgboost4j_spark]$ hadoop fs -mkdir /tmp/xgboost4j_spark
[xgboost4j_spark]$ hadoop fs -copyFromLocal ${SPARK_XGBOOST_DIR}/mortgage/* /tmp/xgboost4j_spark
```

<span id="etl">Launch Mortgage or Taxi ETL Part</span>
---------------------------

Note: the `mortgage_eval_merged.csv` and `mortgage_train_merged.csv` are not Mortgage raw data,
they are the data produced by Mortgage ETL job. If user wants to use a larger size Mortgage data, please refer to [Launch ETL job](#etl).
Taxi ETL job is the same. But Agaricus does not have ETL process, it is combined with XGBoost as there is just a filter operation.

Run spark-submit

``` bash
${SPARK_HOME}/bin/spark-submit \
   --conf spark.plugins=com.nvidia.spark.SQLPlugin \
   --conf spark.rapids.memory.gpu.pooling.enabled=false \
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
   --class com.nvidia.spark.examples.mortgage.ETLMain  \
   $SAMPLE_JAR \
   -format=csv \
   -dataPath="perf::${SPARK_XGBOOST_DIR}/mortgage/perf-train/" \
   -dataPath="acq::${SPARK_XGBOOST_DIR}/mortgage/acq-train/" \
   -dataPath="out::${SPARK_XGBOOST_DIR}/mortgage/out/train/"

# if generating eval data, change the data path to eval as well as the corresponding perf-eval and acq-eval data
# -dataPath="perf::${SPARK_XGBOOST_DIR}/mortgage/perf-eval"
# -dataPath="acq::${SPARK_XGBOOST_DIR}/mortgage/acq-eval"
# -dataPath="out::${SPARK_XGBOOST_DIR}/mortgage/out/eval/"
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
export EXAMPLE_CLASS=com.nvidia.spark.examples.mortgage.GPUMain
# or change to com.nvidia.spark.examples.taxi.GPUMain to run Taxi Xgboost benchmark
# or change to com.nvidia.spark.examples.agaricus.GPUMain to run Agaricus Xgboost benchmark

# tree construction algorithm
export TREE_METHOD=gpu_hist
```

Run spark-submit:

``` bash
${SPARK_HOME}/bin/spark-submit                                                  \
 --conf spark.plugins=com.nvidia.spark.SQLPlugin \
 --conf spark.rapids.memory.gpu.pooling.enabled=false \
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
 -dataPath=train::${DATA_PATH}/mortgage/csv/train/mortgage_train_merged.csv       \
 -dataPath=trans::${DATA_PATH}/mortgage/csv/test/mortgage_eval_merged.csv          \
 -format=csv                                                                    \
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
export EXAMPLE_CLASS=com.nvidia.spark.examples.mortgage.CPUMain
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
 -dataPath=train::${DATA_PATH}/mortgage/csv/train/mortgage_train_merged.csv       \
 -dataPath=trans::${DATA_PATH}/mortgage/csv/test/mortgage_eval_merged.csv          \
 -format=csv                                                                    \
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

Get Started with XGBoost4J-Spark on Apache Hadoop YARN
======================================================
This is a getting started guide to XGBoost4J-Spark on Apache Hadoop YARN supporting GPU scheduling.
At the end of this guide, the reader will be able to run a sample Apache Spark Python application that runs on NVIDIA GPUs.

Prerequisites
-------------

* Apache Spark 3.2.0+ running on YARN supporting GPU scheduling. (e.g.: Spark 3.2.0, Hadoop-Yarn 3.3.0)
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

Please make sure to install the XGBoost, cudf-cu11, numpy libraries on all nodes before running XGBoost application.
``` bash
pip install xgboost
pip install cudf-cu11 --extra-index-url=https://pypi.nvidia.com
pip install numpy
pip install scikit-learn
```
You can also create an isolated python environment by using [Virtualenv](https://virtualenv.pypa.io/en/latest/),
and then directly pass/unpack the archive file and enable the environment on executors
by leveraging the --archives option or spark.archives configuration.
``` bash
# create an isolated python environment and install libraries
python -m venv pyspark_venv
source pyspark_venv/bin/activate
pip install xgboost
pip install cudf-cu11 --extra-index-url=https://pypi.nvidia.com
pip install numpy
pip install scikit-learn
venv-pack -o pyspark_venv.tar.gz

# enable archive python environment on executors
export PYSPARK_DRIVER_PYTHON=python # Do not set in cluster modes.
export PYSPARK_PYTHON=./environment/bin/python
spark-submit --archives pyspark_venv.tar.gz#environment app.py
```

Get Application Files, Jar and Dataset
-------------------------------

Make sure you have prepared the necessary packages and dataset by following this [guide](../prepare-package-data/preparation-python.md)

Then create a directory in HDFS, and run below commands,

``` bash
[xgboost4j_spark_python]$ hadoop fs -mkdir /tmp/xgboost4j_spark_python
[xgboost4j_spark_python]$ hadoop fs -copyFromLocal ${SPARK_XGBOOST_DIR}/mortgage/* /tmp/xgboost4j_spark_python
```

Launch Mortgage or Taxi ETL Part
---------------------------

Use the ETL app to process raw Mortgage data. You can either use this ETLed data to split into training and evaluation data or run the ETL on different subsets of the dataset to produce training and evaluation datasets.

Note: For ETL jobs, Set `spark.task.resource.gpu.amount` to `1/spark.executor.cores`.

``` bash
# location where data was downloaded
export DATA_PATH=hdfs:/tmp/xgboost4j_spark_python/

${SPARK_HOME}/bin/spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --conf spark.executor.cores=10 \
    --conf spark.task.resource.gpu.amount=0.1 \
    --conf spark.rapids.sql.incompatibleDateFormats.enabled=true \
    --conf spark.rapids.sql.csv.read.double.enabled=true \
    --conf spark.sql.cache.serializer=com.nvidia.spark.ParquetCachedBatchSerializer \
    --jars ${RAPIDS_JAR}\
    ${MAIN_PY} \
    --mainClass='com.nvidia.spark.examples.mortgage.etl_main' \
    --format=csv \
    --dataPath="data::${DATA_PATH}/mortgage/data/mortgage/input/" \
    --dataPath="out::${DATA_PATH}/mortgage/data/mortgage/output/train/" \
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
# location where data was downloaded
export DATA_PATH=hdfs:/tmp/xgboost4j_spark_python

# spark deploy mode (see Apache Spark documentation for more information)
export SPARK_DEPLOY_MODE=cluster

# run a single executor for this example to limit the number of spark tasks and
# partitions to 1 as currently this number must match the number of input files
export SPARK_NUM_EXECUTORS=1

# spark driver memory
export SPARK_DRIVER_MEMORY=4g

# spark executor memory
export SPARK_EXECUTOR_MEMORY=8g

# python entrypoint
export SPARK_PYTHON_ENTRYPOINT=${LIBS_PATH}/main.py

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
 --conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh        \
 --files ${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh            \
 --master yarn                                                                  \
 --deploy-mode ${SPARK_DEPLOY_MODE}                                             \
 --archives your_pyspark_venv.tar.gz#environment     #if you enabled archive python environment \
 --num-executors ${SPARK_NUM_EXECUTORS}                                         \
 --driver-memory ${SPARK_DRIVER_MEMORY}                                         \
 --executor-memory ${SPARK_EXECUTOR_MEMORY}                                     \
 --jars ${RAPIDS_JAR}        \
 --py-files ${SAMPLE_ZIP}                   \
 ${MAIN_PY}                                                     \
 --mainClass=${EXAMPLE_CLASS}                                                   \
 --dataPath=train::${DATA_PATH}/mortgage/out/train/      \
 --dataPath=trans::${DATA_PATH}/mortgage/out/eval/        \
 --format=parquet                                                                   \
 --numWorkers=${SPARK_NUM_EXECUTORS}                                            \
 --treeMethod=${TREE_METHOD}                                                    \
 --numRound=100                                                                 \
 --maxDepth=8

# Change the format to csv if your input file is CSV format.
# Please make sure to change the class and data path while running Taxi or Agaricus benchmark  
```

In the `stdout` driver log, you should see timings<sup>*</sup> (in seconds), and the accuracy metric:

```
----------------------------------------------------------------------------------------------------
Training takes 10.75 seconds

----------------------------------------------------------------------------------------------------
Transformation takes 4.38 seconds

----------------------------------------------------------------------------------------------------
Accuracy is 0.997544753891
```

Launch XGBoost Part on CPU
---------------------------

If you are running this example after running the GPU example above, please set these variables, to set both training and testing to run on the CPU exclusively:

``` bash
# location where data was downloaded
export DATA_PATH=hdfs:/tmp/xgboost4j_spark_python/

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
export EXAMPLE_CLASS=com.nvidia.spark.examples.mortgage.main
# or change to com.nvidia.spark.examples.taxi.main to run Taxi Xgboost benchmark
# or change to com.nvidia.spark.examples.agaricus.main to run Agaricus Xgboost benchmark

# tree construction algorithm
export TREE_METHOD=hist

# if you enable archive python environment
export PYSPARK_DRIVER_PYTHON=python
export PYSPARK_PYTHON=./environment/bin/python
```

This is the same command as for the GPU example, repeated for convenience:

``` bash
${SPARK_HOME}/bin/spark-submit                                                  \
 --master yarn                                                                  \
 --archives your_pyspark_venv.tar.gz#environment     #if you enabled archive python environment \
 --deploy-mode ${SPARK_DEPLOY_MODE}                                             \
 --num-executors ${SPARK_NUM_EXECUTORS}                                         \
 --driver-memory ${SPARK_DRIVER_MEMORY}                                         \
 --executor-memory ${SPARK_EXECUTOR_MEMORY}                                     \
 --jars ${RAPIDS_JAR}        \
 --py-files ${SAMPLE_ZIP}                                  \
 ${MAIN_PY}                                                     \
 --mainClass=${EXAMPLE_CLASS}                                                   \
 --dataPath=train::${DATA_PATH}/mortgage/output/train/       \
 --dataPath=trans::${DATA_PATH}/mortgage/output/eval/         \
 --format=parquet                                                               \
 --numWorkers=${SPARK_NUM_EXECUTORS}                                            \
 --treeMethod=${TREE_METHOD}                                                    \
 --numRound=100                                                                 \
 --maxDepth=8
 
 # Please make sure to change the class and data path while running Taxi or Agaricus benchmark  
```

In the `stdout` driver log, you should see timings<sup>*</sup> (in seconds), and the accuracy metric:

```
----------------------------------------------------------------------------------------------------
Training takes 10.76 seconds

----------------------------------------------------------------------------------------------------
Transformation takes 1.25 seconds

----------------------------------------------------------------------------------------------------
Accuracy is 0.998526852335
```

<sup>*</sup> The timings in this Getting Started guide are only for illustrative purpose.
Please see our [release announcement](https://medium.com/rapids-ai/nvidia-gpus-and-apache-spark-one-step-closer-2d99e37ac8fd) for official benchmarks.

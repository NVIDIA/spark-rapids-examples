Get Started with XGBoost4J-Spark on Kubernetes
==============================================
This is a getting started guide to deploy XGBoost4J-Spark package on a Kubernetes cluster. At the end of this guide,
the reader will be able to run a sample Apache Spark XGBoost application on NVIDIA GPU Kubernetes cluster.

Prerequisites
-------------

* Apache Spark 3.2.0+ (e.g.: Spark 3.2.0)
* Hardware Requirements
  * NVIDIA Pascal™ GPU architecture or better
  * Multi-node clusters with homogenous GPU configuration
* Software Requirements
  * Ubuntu 20.04, 22.04/CentOS7, Rocky Linux 8
  * CUDA 11.0+
  * NVIDIA driver compatible with your CUDA
  * NCCL 2.7.8+
* [Kubernetes cluster with NVIDIA GPUs](https://docs.nvidia.com/datacenter/cloud-native/kubernetes/install-k8s.html)
  * See official [Spark on Kubernetes](https://spark.apache.org/docs/latest/running-on-kubernetes.html#prerequisites) 
    instructions for detailed spark-specific cluster requirements
* kubectl installed and configured in the job submission environment
  * Required for managing jobs and retrieving logs

Build a GPU Spark Docker Image
------------------------------

Build a GPU Docker image with Spark resources in it, this Docker image must be accessible by each node in the Kubernetes cluster.

1. Locate your Spark installations. If you don't have one, you can [download](https://spark.apache.org/downloads.html) from Apache and unzip it.
2. `export SPARK_HOME=<path to spark>`
3. [Download the Dockerfile](/dockerfile/Dockerfile) into `${SPARK_HOME}`. (Here CUDA 11.0 is used as an example in the Dockerfile,
   you may need to update it for other CUDA versions.)
4. __(OPTIONAL)__ install any additional library jars into the `${SPARK_HOME}/jars` directory.
    * Most public cloud file systems are not natively supported -- pulling data and jar files from S3, GCS, etc. require installing additional libraries.
5. Build and push the docker image.

``` bash
export SPARK_HOME=<path to spark>
export SPARK_DOCKER_IMAGE=<gpu spark docker image repo and name>
export SPARK_DOCKER_TAG=<spark docker image tag>

pushd ${SPARK_HOME}
wget https://github.com/NVIDIA/spark-rapids-examples/raw/branch-25.06/dockerfile/Dockerfile

# Optionally install additional jars into ${SPARK_HOME}/jars/

docker build . -t ${SPARK_DOCKER_IMAGE}:${SPARK_DOCKER_TAG}
docker push ${SPARK_DOCKER_IMAGE}:${SPARK_DOCKER_TAG}
popd
```

Get Jars and Dataset
-------------------------------

Make sure you have prepared the necessary packages and dataset by following this [guide](/docs/get-started/xgboost-examples/prepare-package-data/preparation-scala.md).

Make sure that data and jars are accessible by each node of the Kubernetes cluster 
via [Kubernetes volumes](https://spark.apache.org/docs/latest/running-on-kubernetes.html#using-kubernetes-volumes), 
on cluster filesystems like HDFS, or in [object stores like S3 and GCS](https://spark.apache.org/docs/2.3.0/cloud-integration.html). 
Note that using [application dependencies](https://spark.apache.org/docs/latest/running-on-kubernetes.html#dependency-management) from 
the submission client’s local file system is currently not yet supported.

#### Note: 
1. Mortgage and Taxi jobs have ETLs to generate the processed data. 
2. For convenience, a subset of [Taxi](/datasets/) dataset is made available in this repo that can be readily used for launching XGBoost job. Use [ETL](#etl) to generate larger datasets for trainig and testing. 
3. Agaricus does not have an ETL process, it is combined with XGBoost as there is just a filter operation.

Save Kubernetes Template Resources
----------------------------------

When using Spark on Kubernetes the driver and executor pods can be launched with pod templates. In the XGBoost4J-Spark use case,
these template yaml files are used to allocate and isolate specific GPUs to each pod. The following is a barebones template file to allocate 1 GPU per pod.

```
apiVersion: v1
kind: Pod
spec:
  containers:
    - name: gpu-example
      resources:
        limits:
          nvidia.com/gpu: 1
```

This 1 GPU template file should be sufficient for all XGBoost jobs because each executor should only run 1 task on a single GPU.
Save this yaml file to the local environment of the machine you are submitting jobs from, 
you will need to provide a path to it as an argument in your spark-submit command. 
Without the template file a pod will see every GPU on the cluster node it is allocated on and can attempt
to execute using a GPU which is already in use -- causing undefined behavior and errors.

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
   --master <k8s://ip:port or k8s://URL>                                                                  \
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
# Variables dependent on how data was made accessible to each node
# Make sure to include relevant spark-submit configuration arguments
# location where data was saved
export DATA_PATH=<path to data directory> 

# Variables independent of how data was made accessible to each node
# kubernetes master URL, used as the spark master for job submission
export SPARK_MASTER=<k8s://ip:port or k8s://URL>

# local path to the template file saved in the previous step
export TEMPLATE_PATH=${HOME}/gpu_executor_template.yaml

# spark docker image location
export SPARK_DOCKER_IMAGE=<spark docker image repo and name>
export SPARK_DOCKER_TAG=<spark docker image tag>

# kubernetes service account to launch the job with
export K8S_ACCOUNT=<kubernetes service account name>

# spark deploy mode, cluster mode recommended for spark on kubernetes
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
${SPARK_HOME}/bin/spark-submit                                                          \
  --conf spark.plugins=com.nvidia.spark.SQLPlugin \
  --conf spark.rapids.memory.gpu.pool=NONE \
  --conf spark.executor.resource.gpu.amount=1 \
  --conf spark.task.resource.gpu.amount=1 \
  --conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
  --files $SPARK_HOME/examples/src/main/scripts/getGpusResources.sh \
  --jars ${RAPIDS_JAR}                           \
  --master ${SPARK_MASTER}                                                              \
  --deploy-mode ${SPARK_DEPLOY_MODE}                                                    \
  --class ${EXAMPLE_CLASS}                                                              \
  --conf spark.executor.instances=${SPARK_NUM_EXECUTORS}                                \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=${K8S_ACCOUNT}         \
  --conf spark.kubernetes.container.image=${SPARK_DOCKER_IMAGE}:${SPARK_DOCKER_TAG}     \
  --conf spark.kubernetes.driver.podTemplateFile=${TEMPLATE_PATH}                       \
  --conf spark.kubernetes.executor.podTemplateFile=${TEMPLATE_PATH}                     \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark                  \
  ${SAMPLE_JAR}                                                                        \
  -dataPath=train::${SPARK_XGBOOST_DIR}/mortgage/output/train/                   \
  -dataPath=trans::${SPARK_XGBOOST_DIR}/mortgage/output/eval/                    \
  -format=parquet                                                                \
  -numWorkers=${SPARK_NUM_EXECUTORS}                                                    \
  -treeMethod=${TREE_METHOD}                                                            \
  -numRound=100                                                                         \
  -maxDepth=8                   
  
   # Please make sure to change the class and data path while running Taxi or Agaricus benchmark                                                       
                                                
```

Retrieve the logs using the driver's pod name that is printed to `stdout` by spark-submit 
```
export POD_NAME=<kubernetes pod name>
kubectl logs -f ${POD_NAME}
```

In the driver log, you should see timings* (in seconds), and the accuracy metric(take Mortgage as example):
```
--------------
==> Benchmark: Elapsed time for [Mortgage GPU train csv stub Unknown Unknown Unknown]: 30.132s
--------------

--------------
==> Benchmark: Elapsed time for [Mortgage GPU transform csv stub Unknown Unknown Unknown]: 22.352s
--------------

--------------
==> Benchmark: Accuracy for [Mortgage GPU Accuracy csv stub Unknown Unknown Unknown]: 0.9869451418401349
--------------
```

\* Kubernetes logs may not be nicely formatted since `stdout` and `stderr` are not kept separately.

\* The timings in this Getting Started guide are only for illustrative purpose. 
Please see our [release announcement](https://medium.com/rapids-ai/nvidia-gpus-and-apache-spark-one-step-closer-2d99e37ac8fd) for official benchmarks.

# Microbenchmark

## Introduction

The benchmark on [RAPIDS Accelerator For Apache Spark](https://nvidia.github.io/spark-rapids/) is to time the
conversion of incoming RDDs to and from a CachedBatch. Specifically to compare the performance of
ParquetCachedBatchSerializer to DefaultCachedBatchSerializer.

## Setup

### Build Conventional Dist Rapids Plugin jar

Clone the repo and build the jar with following commands:
``` 
git clone https://github.com/NVIDIA/spark-rapids
cd spark-rapids
mvn clean install -DskipTests -DallowConventionalDistJar -Dbuildver=312
```
find the jar file `rapids-4-spark_2.12-version-cuda11.jar` under path spark-rapids/dist/target
Note : the benchmark is specific to spark 3.1.x, the guide is based on spark3.1.2
### Build Toree kernel

Refer to the [doc](/docs/get-started/xgboost-examples/notebook/toree.md) to build the 
'toree' locally to support scala 2.12, and install it.
Install a new kernel with the jar you build.
```
export SPARK_HOME=your-spark-home
export PYSPARK_DRIVER_PYTHON=your-spark-home
export RAPIDS_JAR=jar-path-to-your-build
export SPARK_MASTER=your-url

NUM_EXECUTOR_CORES=4
EXECUTOR_MEMORY=4
DRIVER_MEMORY=4

jupyter toree install                                \
--spark_home=${SPARK_HOME}                             \
--user  \
--debug \
--log-level="DEBUG" \
--toree_opts="--nosparkcontext"                         \
--kernel_name="pcbs"                         \
--spark_opts="--master ${SPARK_MASTER} \
  --jars ${RAPIDS_JAR}      \
  --driver-memory ${DRIVER_MEMORY}G \
  --executor-cores $NUM_EXECUTOR_CORES \
  --executor-memory ${EXECUTOR_MEMORY}G \
  --conf spark.rapids.sql.enabled=true \
  --conf spark.rapids.sql.explain=ALL \
  --conf spark.plugins=com.nvidia.spark.SQLPlugin  \
  --conf spark.driver.extraClassPath=${RAPIDS_JAR} \
  --conf spark.executor.extraClassPath=${RAPIDS_JAR} \
  --conf spark.rapids.memory.gpu.pooling.enabled=false \
  --conf spark.executor.resource.gpu.amount=1 \
  --conf spark.task.resource.gpu.amount=0.25 \
  --conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
  --files $SPARK_HOME/examples/src/main/scripts/getGpusResources.sh"
```
### Run Benchmark

``` 
jupyter notebook
``` 
after you type the above command, a web page will be open, click the `PCBS-benchmark.ipynb` file and
changed the kernel to `pcbs-scala`, then run the notebook.

You can find the output and get the time cost information in the last output cell.
The performance can be different as the cache data size changes, below are some results for reference:
![run-pcbs-benchmark](/docs/img/guides/pcbs-perf.png)

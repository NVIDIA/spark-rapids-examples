# Spark-Rapids-ML PCA example

This is an example of the GPU accelerated PCA algorithm from the [Spark-Rapids-ML](https://github.com/NVIDIA/spark-rapids-ml) library, which provides PySpark ML compatible algorithms powered by RAPIDS cuML. 
The notebook uses PCA to reduce a random dataset with 2048 feature dimensions to 3 dimensions. We train both the GPU and CPU algorithms for comparison. 

## Build

Please refer to the Spark-Rapids-ML [README](https://github.com/NVIDIA/spark-rapids-ml/blob/HEAD/python) for Python build instructions and API usage.

## Running the Notebooks

Once you have built your environment, please follow these instructions to run the notebooks.

**Note**: for demonstration purposes, these examples just use a local Spark Standalone cluster with a single executor, but you should be able to run them on any distributed Spark cluster.
```
# setup environment variables
export SPARK_HOME=/path/to/spark
export MASTER=spark://$(hostname):7077
export SPARK_WORKER_INSTANCES=1
export CORES_PER_WORKER=8
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS='lab'

# start spark standalone cluster
${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-worker.sh -c ${CORES_PER_WORKER} -m 16G ${MASTER}

# start jupyter with pyspark
${SPARK_HOME}/bin/pyspark --master ${MASTER} \
--driver-memory 10G \
--executor-memory 8G \
--conf spark.python.worker.reuse=True

# BROWSE to localhost:8888 to view/run notebooks

# stop spark standalone cluster
${SPARK_HOME}/sbin/stop-worker.sh; ${SPARK_HOME}/sbin/stop-master.sh
```

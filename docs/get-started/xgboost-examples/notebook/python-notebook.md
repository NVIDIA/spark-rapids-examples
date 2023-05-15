Get Started with pyspark+XGBoost with Jupyter Notebook
===================================================================

This is a getting started guide to XGBoost4J-Spark using an [Jupyter notebook](https://jupyter.org/). 
At the end of this guide, you will be able to run a sample notebook that runs on NVIDIA GPUs.

Before you begin, please ensure that you have setup a Spark Cluster(Standalone or YARN).
You should change `--master` config according to your cluster architecture. For example, set `--master yarn` for spark on YARN.

It is assumed that the `SPARK_MASTER` and `SPARK_HOME` environment variables are defined and point to the Spark Master URL (e.g. `spark://localhost:7077`),
and the home directory for Apache Spark respectively.

1. Make sure you have [Jupyter notebook installed](https://jupyter.org/install.html).

   If you install it with conda, please make sure your Python version is consistent.

2. Prepare packages and dataset.

    Make sure you have prepared the necessary packages and dataset by following this [guide](../prepare-package-data/preparation-python.md)

3. Launch the notebook:

   Note: For ETL jobs, Set `spark.task.resource.gpu.amount` to `1/spark.executor.cores`.

    For ETL:

    ``` bash
    PYSPARK_DRIVER_PYTHON=jupyter       \
    PYSPARK_DRIVER_PYTHON_OPTS=notebook \
    pyspark                             \
    --master ${SPARK_MASTER}            \
    --jars ${RAPIDS_JAR}\
    --py-files ${SAMPLE_ZIP}      \
    --conf spark.plugins=com.nvidia.spark.SQLPlugin \
    --conf spark.executor.resource.gpu.amount=1 \
    --conf spark.executor.cores=10 \
    --conf spark.task.resource.gpu.amount=0.1 \
    --conf spark.sql.cache.serializer=com.nvidia.spark.ParquetCachedBatchSerializer \
    --conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
    --files $SPARK_HOME/examples/src/main/scripts/getGpusResources.sh
    ```

    For XGBoost:

    ``` bash
    PYSPARK_DRIVER_PYTHON=jupyter       \
    PYSPARK_DRIVER_PYTHON_OPTS=notebook \
    pyspark                             \
    --master ${SPARK_MASTER}            \
    --jars ${RAPIDS_JAR}\
    --py-files ${SAMPLE_ZIP}      \
    --conf spark.plugins=com.nvidia.spark.SQLPlugin \
    --conf spark.rapids.memory.gpu.pool=NONE \
    --conf spark.executor.resource.gpu.amount=1 \
    --conf spark.executor.cores=10 \
    --conf spark.task.resource.gpu.amount=1 \
    --conf spark.sql.execution.arrow.maxRecordsPerBatch=200000 \
    --conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
    --files $SPARK_HOME/examples/src/main/scripts/getGpusResources.sh
    ```

4. Launch ETL Part 

- Mortgage ETL Notebook: [Python](../../../../examples/XGBoost-Examples/mortgage/notebooks/python/MortgageETL.ipynb)
- Taxi ETL Notebook: [Python](../../../../examples/XGBoost-Examples/taxi/notebooks/python/taxi-ETL.ipynb)
- Note: Agaricus does not have ETL part.

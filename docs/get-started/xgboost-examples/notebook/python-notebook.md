Get Started with XGBoost4J-Spark with Jupyter Notebook
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

    Make sure you have prepared the necessary packages and dataset by following this [guide](/docs/get-started/xgboost-examples/prepare-package-data/preparation-python.md)

3. Launch the notebook:

    ``` bash
    PYSPARK_DRIVER_PYTHON=jupyter       \
    PYSPARK_DRIVER_PYTHON_OPTS=notebook \
    pyspark                             \
    --master ${SPARK_MASTER}            \
    --conf spark.executor.extraClassPath=${CUDF_JAR}:${RAPIDS_JAR} \
    --jars ${CUDF_JAR},${RAPIDS_JAR},${XGBOOST4J_JAR},${XGBOOST4J_SPARK_JAR}\
    --py-files ${XGBOOST4J_SPARK_JAR},${SAMPLE_ZIP}      \
    --conf spark.plugins=com.nvidia.spark.SQLPlugin \
    --conf spark.rapids.memory.gpu.pooling.enabled=false \
    --conf spark.executor.resource.gpu.amount=1 \
    --conf spark.task.resource.gpu.amount=1 \
    --conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
    --files $SPARK_HOME/examples/src/main/scripts/getGpusResources.sh
    ```

4. Launch ETL Part 
- Mortgage ETL Notebook: [Python](/examples/Spark-ETL+XGBoost(splitting)/mortgage/notebooks/python/MortgageETL.ipynb)
- Taxi ETL Notebook: [Python](/examples/Spark-ETL+XGBoost(splitting)/taxi/notebooks/python/taxi-ETL.ipynb)
- Note: Agaricus does not have ETL part.
   
5. Launch XGBoost Part
- Mortgage XGBoost Notebook: [Python](/examples/Spark-ETL+XGBoost(splitting)/mortgage/notebooks/python/mortgage-gpu.ipynb)
- Taxi XGBoost Notebook: [Python](/examples/Spark-ETL+XGBoost(splitting)/taxi/notebooks/python/taxi-gpu.ipynb)
- Agaricus XGBoost Notebook: [Python](/examples/Spark-ETL+XGBoost(splitting)/agaricus/notebooks/python/agaricus-gpu.ipynb)
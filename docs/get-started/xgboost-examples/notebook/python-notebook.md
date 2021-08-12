Get Started with XGBoost4J-Spark with Jupyter Notebook
===================================================================

This is a getting started guide to XGBoost4J-Spark using an [Jupyter notebook](https://jupyter.org/). At the end of this guide, the reader will be able to run a sample notebook that runs on NVIDIA GPUs.

Before you begin, please ensure that you have setup a [Spark Standalone Cluster](/docs/get-started/xgboost-examples/on-prem-cluster/standalone-python.md).

It is assumed that the `SPARK_MASTER` and `SPARK_HOME` environment variables are defined and point to the master spark URL (e.g. `spark://localhost:7077`), and the home directory for Apache Spark respectively.

1. Make sure you have [Jupyter notebook installed](https://jupyter.org/install.html).

   If you install it with conda, please makes sure your Python version is consistent.

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

    Then you start your notebook and open [`mortgage-gpu.ipynb`](/examples/mortgage/notebooks/scala/mortgage-gpu.ipynb) to explore.

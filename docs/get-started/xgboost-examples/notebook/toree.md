Get Started with XGBoost4J-Spark with Apache Toree Jupyter Notebook
===================================================================

This is a getting started guide to XGBoost4J-Spark using an [Apache Toree](https://toree.apache.org/) Jupyter notebook. 
At the end of this guide, you will be able to run a sample notebook that runs on NVIDIA GPUs.

Before you begin, please ensure that you have setup a Spark Cluster(Standalone or YARN).
You should change `--master` config according to your cluster architecture. For example, set `--master yarn` for spark on YARN.

It is assumed that the `SPARK_MASTER` and `SPARK_HOME` environment variables are defined and point to the master spark URL (e.g. `spark://localhost:7077`),
and the home directory for Apache Spark respectively.

1. Make sure you have jupyter notebook and [sbt](https://www.scala-sbt.org/1.x/docs/Installing-sbt-on-Linux.html) installed first.
2. Build the 'toree' locally to support scala 2.12, and install it.

    ``` bash
    # Download toree
    wget https://github.com/apache/incubator-toree/archive/refs/tags/v0.5.0-incubating-rc4.tar.gz
    tar -xvzf v0.5.0-incubating-rc4.tar.gz
    # Build the Toree pip package.
    cd incubator-toree-0.5.0-incubating-rc4
    make pip-release
    # Install Toree
    pip install dist/toree-pip/toree-0.5.0.tar.gz
    ```
3. Prepare packages and dataset.

    Make sure you have prepared the necessary packages and dataset by following this [guide](/docs/get-started/xgboost-examples/prepare-package-data/preparation-scala.md)

4. Install a new kernel with gpu enabled and lunch

    ``` bash
    jupyter toree install                                \
    --spark_home=${SPARK_HOME}                             \
    --user                                          \
    --toree_opts='--nosparkcontext'                         \
    --kernel_name="XGBoost4j-Spark"                         \
    --spark_opts='--master ${SPARK_MASTER} \
      --jars ${CUDF_JAR},${RAPIDS_JAR},${SAMPLE_JAR}       \
      --conf spark.plugins=com.nvidia.spark.SQLPlugin  \
      --conf spark.executor.extraClassPath=${CUDF_JAR}:${RAPIDS_JAR} \
      --conf spark.rapids.memory.gpu.pooling.enabled=false \
      --conf spark.executor.resource.gpu.amount=1 \
      --conf spark.task.resource.gpu.amount=1 \
      --conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
      --files $SPARK_HOME/examples/src/main/scripts/getGpusResources.sh'
    ```

    Launch the notebook:

    ``` bash
    jupyter notebook
    ```

4. Launch ETL Part 
- Mortgage ETL Notebook: [Scala](/examples/Spark-ETL+XGBoost/mortgage/notebooks/scala/mortgage-ETL.ipynb)
- Taxi ETL Notebook: [Scala](/examples/Spark-ETL+XGBoost/taxi/notebooks/scala/taxi-ETL.ipynb)
- Note: Agaricus does not have ETL part.
   
5. Launch XGBoost Part
- Mortgage XGBoost Notebook: [Scala](/examples/Spark-ETL+XGBoost/mortgage/notebooks/scala/mortgage-gpu.ipynb)
- Taxi XGBoost Notebook: [Scala](/examples/Spark-ETL+XGBoost/taxi/notebooks/scala/taxi-gpu.ipynb)
- Agaricus XGBoost Notebook: [Scala](/examples/Spark-ETL+XGBoost/agaricus/notebooks/scala/agaricus-gpu.ipynb)
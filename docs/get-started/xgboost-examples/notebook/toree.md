Get Started with XGBoost4J-Spark with Apache Toree Jupyter Notebook
===================================================================

This is a getting started guide to XGBoost4J-Spark using an [Apache Toree](https://toree.apache.org/) Jupyter notebook. 
At the end of this guide, the reader will be able to run a sample notebook that runs on NVIDIA GPUs.

Before you begin, please ensure that you have setup a Spark Cluster no matter standalone or yarn, the only difference is that
you should change `--master` param according to your cluster deploy mode, if the deploy mode is yarn, you should set `--master yarn`.

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

5. Launch Mortgage or Taxi ETL Part 
   Run [Mortgage ETL Notebook](../../../../examples/Spark-ETL+XGBoost/mortgage/notebooks/scala/mortgage-ETL.ipynb) to process ETL part.
   Taxi ETL job is the same, just change to [Taxi ETL Notebook](../../../../examples/Spark-ETL+XGBoost/taxi/notebooks/scala/taxi-ETL.ipynb).
   But Agaricus does not have ETL process, it is combined with XGBoost as there is just a filter operation.
   
6. Launch XGBoost Part on GPU/CPU
   Run [Mortgage XGBoost Notebook](../../../../examples/Spark-ETL+XGBoost/mortgage/notebooks/scala/mortgage-gpu.ipynb) to process XGBoost part.
   Taxi and Agaricus ETL job and are the same, just change to [Taxi XGBoost Notebook](../../../../examples/Spark-ETL+XGBoost/taxi/notebooks/scala/taxi-gpu.ipynb)
   or [Agaricus Notebook](../../../../examples/Spark-ETL+XGBoost/agaricus/notebooks/scala/agaricus-gpu.ipynb)

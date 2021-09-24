# PCA example

This is an example of the GPU accelerated PCA algorithm running on Spark.

## Build

Please refer to [README](https://github.com/NVIDIA/spark-rapids-ml#readme) in the [spark-rapids-ml](https://github.com/NVIDIA/spark-rapids-ml) github repository for build instructions and API usage.

## Sample code

User can find sample scala code in [`main.scala`](./main.scala). In the sample code, we will generate random data with 2048 feature dimensions. Then we use PCA to reduce number of features to 3.

Just copy the sample code into the spark-shell laucnhed according to [this section](https://github.com/NVIDIA/spark-rapids-ml#how-to-use) and REPL will give out the algorithm results.

## Notebook

[Apache Toree](https://toree.apache.org/) is required to run PCA sample code in a Jupyter Notebook. 

It is assumed that a Standalone Spark cluster has been set up, the `SPARK_MASTER` and `SPARK_HOME` environment variables are defined and point to the master spark URL (e.g. `spark://localhost:7077`), and the home directory for Apache Spark respectively.

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

3. Install a new kernel with the jar(use $RAPIDS_ML_JAR for reference) built from section [Build](#build) and launch

    ``` bash
    jupyter toree install                                \
    --spark_home=${SPARK_HOME}                             \
    --user                                          \
    --toree_opts='--nosparkcontext'                         \
    --kernel_name="spark-rapids-ml-pca"                         \
    --spark_opts='--master ${SPARK_MASTER} \
      --jars ${RAPIDS_ML_JAR}       \
      --conf spark.driver.memory=10G \
      --conf spark.executor.memory=10G \
      --conf spark.executor.heartbeatInterval=20s \
      --conf spark.executor.extraClassPath=${RAPIDS_ML_JAR} \
      --conf spark.executor.resource.gpu.amount=1 \
      --conf spark.task.resource.gpu.amount=1 \
      --conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
      --files $SPARK_HOME/examples/src/main/scripts/getGpusResources.sh'
    ```

    Launch the notebook:

    ``` bash
    jupyter notebook
    ```

    Please choose "spark-rapids-ml-pca" as your notebook kernel.
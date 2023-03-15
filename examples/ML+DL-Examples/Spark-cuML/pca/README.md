# PCA example

This is an example of the GPU accelerated PCA algorithm running on Spark.

## Build

Please refer to [README](https://github.com/NVIDIA/spark-rapids-ml#readme) in the [spark-rapids-ml](https://github.com/NVIDIA/spark-rapids-ml) github repository for build instructions and API usage.

## Get jars from Maven Central

User can also download the release jar from Maven central:

[rapids-4-spark-ml_2.12-22.02.0-cuda11.jar](https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark-ml_2.12/22.02.0/rapids-4-spark-ml_2.12-22.02.0-cuda11.jar)

[rapids-4-spark_2.12-23.02.0.jar](https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/23.02.0/rapids-4-spark_2.12-23.02.0.jar)


## Sample code

User can find sample scala code in [`main.scala`](main.scala). In the sample code, we will generate random data with 2048 feature dimensions. Then we use PCA to reduce number of features to 3.

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
    RAPIDS_ML_JAR=PATH_TO_rapids-4-spark-ml_2.12-22.02.0-cuda11.jar
    PLUGIN_JAR=PATH_TO_rapids-4-spark_2.12-23.04.0-SNAPSHOT.jar

    jupyter toree install                                \
    --spark_home=${SPARK_HOME}                             \
    --user                                          \
    --toree_opts='--nosparkcontext'                         \
    --kernel_name="spark-rapids-ml-pca"                         \
    --spark_opts='--master ${SPARK_MASTER} \
      --jars ${RAPIDS_ML_JAR},${PLUGIN_JAR}       \
      --conf spark.driver.memory=10G \
      --conf spark.executor.memory=10G \
      --conf spark.executor.heartbeatInterval=20s \
      --conf spark.rapids.sql.enabled=true \
      --conf spark.plugins=com.nvidia.spark.SQLPlugin \
      --conf spark.rapids.memory.gpu.allocFraction=0.35 \
      --conf spark.rapids.memory.gpu.maxAllocFraction=0.6 \
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



## Submit app jar

We also provide the spark-submit way to run the PCA example. We suggest using Dockerfile to get a clean runnning environment:

```bash
docker build -f Dockerfile -t nvspark/pca:0.1 .
```
Then get into the container of this image(`nvidia-docker` is required as we will use GPU then):
```bash
nvidia-docker run -it nvspark/pca:0.1 bash
```

In this docker image, we assume that user has 2 GPUs in his machine. If you are not the condition, please modify the `-Dspark.worker.resource.gpu.amount` in `spark-env.sh` according to your actual environment.

Then just start the standalone Spark and submit the job:
``` bash
./start-spark.sh
./spark-submit.sh
```

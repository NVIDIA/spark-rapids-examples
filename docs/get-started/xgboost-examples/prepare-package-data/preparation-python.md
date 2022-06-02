## Prepare packages and dataset for pyspark

For simplicity export the location to these jars. All examples assume the packages and dataset will be placed in the `/opt/xgboost` directory:

### Download the jars

1. Download the XGBoost for Apache Spark jars
   * [XGBoost4j Package](https://repo1.maven.org/maven2/ml/dmlc/xgboost4j-gpu_2.12/1.6.1/)
   * [XGBoost4j-Spark Package](https://repo1.maven.org/maven2/ml/dmlc/xgboost4j-spark-gpu_2.12/1.6.1/)

2. Download the RAPIDS Accelerator for Apache Spark plugin jar
   * [RAPIDS Spark Package](https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/22.06.0/rapids-4-spark_2.12-22.06.0.jar)

### Build XGBoost Python Examples

Following this [guide](../../../../docs/get-started/xgboost-examples/building-sample-apps/python.md), you can get *samples.zip* and *main.py* and copy them to `/opt/xgboost`

### Download dataset

You need to download Mortgage dataset to `/opt/xgboost` from this [site](https://docs.rapids.ai/datasets/mortgage-data)
, download Taxi dataset from this [site](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
, download Agaricus dataset from this [site](https://gust.dev/r/xgboost-agaricus).

### Setup environments

``` bash
export SPARK_XGBOOST_DIR=/opt/xgboost
export RAPIDS_JAR=${SPARK_XGBOOST_DIR}/rapids-4-spark_2.12-22.06.0.jar
export XGBOOST4J_JAR=${SPARK_XGBOOST_DIR}/xgboost4j-gpu_2.12-1.6.1.jar
export XGBOOST4J_SPARK_JAR=${SPARK_XGBOOST_DIR}/xgboost4j-spark-gpu_2.12-1.6.1.jar
export SAMPLE_ZIP=${SPARK_XGBOOST_DIR}/samples.zip
export MAIN_PY=${SPARK_XGBOOST_DIR}/main.py
```

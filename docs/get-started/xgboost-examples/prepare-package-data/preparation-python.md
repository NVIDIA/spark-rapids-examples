## Prepare packages and dataset for pyspark

For simplicity export the location to these jars. All examples assume the packages and dataset will be placed in the `/opt/xgboost` directory:

### Download the jars

1. Download the XGBoost for Apache Spark jars
   * [XGBoost4j Package](https://repo1.maven.org/maven2/com/nvidia/xgboost4j_3.0/1.4.2-0.3.0/)
   * [XGBoost4j-Spark Package](https://repo1.maven.org/maven2/com/nvidia/xgboost4j-spark_3.0/1.4.2-0.3.0/)

2. Download the RAPIDS Accelerator for Apache Spark plugin jar
   * [RAPIDS Spark Package](https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/22.10.0/rapids-4-spark_2.12-22.10.0.jar)

### Build XGBoost Python Examples

Following this [guide](/docs/get-started/xgboost-examples/building-sample-apps/python.md), you can get *samples.zip* and *main.py* and copy them to `/opt/xgboost`

### Download dataset

You need to copy the dataset to `/opt/xgboost`. Use the following links to download the data.
1. [Mortgage dataset](/docs/get-started/xgboost-examples/dataset/mortgage.md)
2. [Taxi dataset](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
3. [Agaricus dataset](https://gust.dev/r/xgboost-agaricus)

### Setup environments

``` bash
export SPARK_XGBOOST_DIR=/opt/xgboost
export RAPIDS_JAR=${SPARK_XGBOOST_DIR}/rapids-4-spark_2.12-22.10.0.jar
export XGBOOST4J_JAR=${SPARK_XGBOOST_DIR}/xgboost4j_3.0-1.4.2-0.3.0.jar
export XGBOOST4J_SPARK_JAR=${SPARK_XGBOOST_DIR}/xgboost4j-spark_3.0-1.4.2-0.3.0.jar
export SAMPLE_ZIP=${SPARK_XGBOOST_DIR}/samples.zip
export MAIN_PY=${SPARK_XGBOOST_DIR}/main.py
```

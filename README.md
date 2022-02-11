# spark-rapids-examples

A repo for Spark related utilities and examples using the Rapids Accelerator,including ETL, ML/DL, etc.

Enterprise AI is built on ETL pipelines and relies on AI infrastructure to effectively integrate and
process large amounts of data. One of the fundamental purposes of
[RAPIDS Accelerator](https://nvidia.github.io/spark-rapids/Getting-Started/)
is to effectively integrate large ETL and ML/DL pipelines. Rapids Accelerator for [Apache Spark](https://spark.apache.org/)
offers seamless integration with Machine learning frameworks such XGBoost, PCA. Users can leverage the Apache Spark cluster
with NVIDIA GPUs to accelerate the ETL pipelines and then use the same infrastructure to load the data frame
into single or multiple GPUs across multiple nodes to train with GPU accelerated XGBoost or a PCA.
In addition, if you are using a Deep learning framework to train your tabular data with the same Apache Spark cluster,
we have leveraged NVIDIA’s NVTabular library to load and train the data across multiple nodes with GPUs.
NVTabular is a feature engineering and preprocessing library for tabular data designed to quickly and
easily manipulate terabyte scale datasets used to train deep learning based recommender systems.
We also add MIG support to YARN to allow CSPs to split an A100/A30 into multiple MIG
devices and have them appear like a normal GPU.

Please see the [Rapids Accelerator for Spark documentation](https://nvidia.github.io/spark-rapids/Getting-Started/) for supported
Spark versions and requirements. It is recommended to set up Spark Cluster with JDK8.

## Getting Started Guides

### 1. Microbenchmark guide

The microbenchmark on [RAPIDS Accelerator For Apache Spark](https://nvidia.github.io/spark-rapids/) is to identify,
test and analyze the best queries which can be accelerated on the GPU. For detail information please refer to this
[guide](/examples/micro-benchmarks/README.md).

### 2. Xgboost examples guide

We provide three similar Xgboost benchmarks, Mortgage, Taxi and Agaricus.
Try one of the ["Getting Started Guides"](/examples/Spark-ETL+XGBoost/README.md).
Please note that they target the Mortgage dataset as written with a few changes
to `EXAMPLE_CLASS` and `dataPath`, they can be easily adapted with each other with different datasets.

### 3. TensorFlow training on Horovod Spark example guide

We provide a Criteo Benchmark to demo ETL and deep learning training on Horovod Spark, please refer to
this [guide](/examples/Spark-DL/criteo_train/README.md).

### 4. PCA example guide

This is an example of the GPU accelerated PCA algorithm running on Spark. For detail information please refer to this
[guide](/examples/Spark-cuML/pca/README.md).

### 5. MIG support
We provide some [guides](/examples/MIG-Support/README.md) about the Multi-Instance GPU (MIG) feature based on
the NVIDIA Ampere architecture (such as NVIDIA A100 and A30) GPU.

### 6. Spark Rapids UDF examples
This is examples of the GPU accelerated UDF.
refer to this
[guide](/examples/RAPIDS-accelerated-UDFs/README.md).

## API
### 1. Xgboost examples API

These guides focus on GPU related Scala and python API interfaces.
- [Scala API](/docs/api-docs/xgboost-examples-api-docs/scala.md)
- [Python API](/docs/api-docs/xgboost-examples-api-docs/python.md)

## Troubleshooting
You can trouble-shooting issues according to following guides.
- [Trouble Shooting XGBoost](/docs/trouble-shooting/xgboost-examples-trouble-shooting.md)

## Contributing
See the [Contributing guide](CONTRIBUTING.md).

## Contact Us

Please see the [RAPIDS](https://rapids.ai/community.html) website for contact information.

## License

This content is licensed under the [Apache License 2.0](/LICENSE)

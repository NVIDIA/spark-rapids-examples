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

### 1. SQL & DF Examples

#### 1.1 Microbenchmark example
The microbenchmark on [RAPIDS Accelerator For Apache Spark](https://nvidia.github.io/spark-rapids/) is to identify,
test and analyze the best queries which can be accelerated on the GPU. Rapids Accelerator for Apache Spark reaps 
the significant benefit of GPU performance for some specific operations like Expand, Aggregate, Windowing, Crossjoin, etc.
![microbenchmark-speedup](docs/img/guides/microbenchmark-speedups.png)

For detail information please refer to this
[guide](examples/SQL+DF-Examples/micro-benchmarks).

#### 1.2 Customer Churn example
This demo shows a realistic ETL workflow based on synthetic normalized data. 
For detail information please refer to this
[guide](examples/SQL+DF-Examples/customer-churn).

### 2. Xgboost examples

We provide three similar Xgboost benchmarks, Mortgage, Taxi and Agaricus.
Try one of the ["Getting Started Guides"](examples/XGBoost-Examples).
Please note that they target the Mortgage dataset as written with a few changes
to `EXAMPLE_CLASS` and `dataPath`, they can be easily adapted with each other with different datasets.
Below is the performance benefit with 680 GB input data size on an 8 nodes Spark standalone cluster 
which with 128 CPU cores and 1 A100 GPU on each node.
![mortgage-speedup](docs/img/guides/mortgage-speedups.png)

### 3. Machine Learning & Deep Learning examples

#### 3.1 TensorFlow training on Horovod Spark example
We provide a Criteo Benchmark to demo ETL and deep learning training on Horovod Spark, please refer to
this [guide](examples/ML+DL-Examples/Spark-DL/criteo_train).

#### 3.2 PCA example
This is an example of the GPU accelerated PCA algorithm running on Spark. For detail information please refer to this
[guide](examples/ML+DL-Examples/Spark-cuML/pca).

### 4. UDF Related examples

#### 4.1 Spark Rapids UDF examples
This is examples of the GPU accelerated UDF.
refer to this
[guide](examples/UDF-Examples/RAPIDS-accelerated-UDFs).

#### 4.2 Spark cuSpatial
This is a RapidsUDF examples to use [cuSpatial](https://github.com/rapidsai/cuspatial) library to
solve the point-in-polygon problem. For detail information please refer to this [guide](examples/UDF-Examples/Spark-cuSpatial).

### 5. MIG support
We provide some [guides](examples/MIG-Support) about the Multi-Instance GPU (MIG) feature based on
the NVIDIA Ampere architecture (such as NVIDIA A100 and A30) GPU.

## API
### 1. Xgboost examples API

These guides focus on GPU related Scala and python API interfaces.
- [Scala API](docs/api-docs/xgboost-examples-api-docs/scala.md)
- [Python API](docs/api-docs/xgboost-examples-api-docs/python.md)

## Troubleshooting
You can trouble-shooting issues according to following guides.
- [Trouble Shooting XGBoost](docs/trouble-shooting/xgboost-examples-trouble-shooting.md)

## Contributing
See the [Contributing guide](CONTRIBUTING.md).

## Contact Us

Please see the [RAPIDS](https://rapids.ai/community.html) website for contact information.

## License

This content is licensed under the [Apache License 2.0](LICENSE)
